# Advanced Features (Phase 10)

핵심 개념과 경계에서 Truthound, Phase, Module, Lineage, Realtime을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. 핵심 개념과 경계에서 Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - [이상치 Detection](#anomaly-detection)
   - [드리프트 Detection](#drift-detection)
   - 핵심 개념과 경계에서 Rule, Learning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - 핵심 개념과 경계에서 Model, Registry을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 Lineage, Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - 핵심 개념과 경계에서 Lineage, Graph을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - 핵심 개념과 경계에서 Lineage, Tracker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - 핵심 개념과 경계에서 Impact, Analysis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 Realtime, Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - 핵심 개념과 경계에서 Streaming, Sources을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - [Incremental 검증](#incremental-validation)
   - 핵심 개념과 경계에서 State, Management을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
   - [체크포인트ing](#checkpointing)
5. 핵심 개념과 경계에서 CLI, Commands을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. [API 레퍼런스](#api-reference)

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

핵심 개념과 경계에서 Truthound, Phase을(를) 다루는 항목입니다:

| 핵심 개념과 경계에서 Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 주요 기능 |
|--------|---------|--------------|
| 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Machine learning-based 데이터 품질 | 핵심 개념과 경계에서 Anomaly을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Lineage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Data 플로우 tracking | 핵심 개념과 경계에서 Graph-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Realtime을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 검증 | Incremental 검증, checkpointing |

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ML Module

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 이상치 Detection

핵심 개념과 경계에서 Detect, ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 핵심 개념과 경계에서 `LazyFrame`, Note, LazyFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml import ZScoreAnomalyDetector, IsolationForestDetector, EnsembleAnomalyDetector
import polars as pl

# Sample data - use LazyFrame with at least 10 samples
df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 2.5, 1.5, 2.8, 1.8, 2.2, 3.1, 2.9, 100.0],  # 100.0 is an anomaly
}).lazy()

# Import specific configs
from truthound.ml.anomaly_models.statistical import StatisticalConfig
from truthound.ml.anomaly_models.isolation_forest import IsolationForestConfig
from truthound.ml.anomaly_models.ensemble import EnsembleConfig, EnsembleStrategy

# Z-Score based detection (use StatisticalConfig)
zscore_detector = ZScoreAnomalyDetector(
    config=StatisticalConfig(z_threshold=3.0)
)
zscore_detector.fit(df)
result = zscore_detector.predict(df)
print(f"Anomalies found: {result.anomaly_count}")

# Isolation Forest (ML-based, use IsolationForestConfig)
iso_detector = IsolationForestDetector(
    config=IsolationForestConfig(contamination=0.1)
)
iso_detector.fit(df)
result = iso_detector.predict(df)

# Ensemble approach (use EnsembleConfig with EnsembleStrategy)
ensemble = EnsembleAnomalyDetector(
    detectors=[zscore_detector, iso_detector],
    config=EnsembleConfig(strategy=EnsembleStrategy.AVERAGE)
)
ensemble.fit(df)
result = ensemble.predict(df)
```

#### Available Detectors

| 핵심 개념과 경계에서 Detector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|--------|----------|
| 핵심 개념과 경계에서 `ZScoreAnomalyDetector`, ZScoreAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Z-Score을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Normally을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `IQRAnomalyDetector`, IQRAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Interquartile, Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Skewed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `MADAnomalyDetector`, MADAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Median, Absolute, Deviation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Robust을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `IsolationForestDetector`, IsolationForestDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 High-dimensional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `EnsembleAnomalyDetector`, EnsembleAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Voting을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Combining을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 드리프트 Detection

핵심 개념과 경계에서 Detect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 핵심 개념과 경계에서 `LazyFrame`, Note, Drift, LazyFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml import DistributionDriftDetector, FeatureDriftDetector
import polars as pl

# Baseline and current data - use LazyFrame
baseline = pl.DataFrame({"value": [1, 2, 3, 4, 5] * 20}).lazy()
current = pl.DataFrame({"value": [10, 11, 12, 13, 14] * 20}).lazy()  # Drifted!

# Import specific configs for drift detectors
from truthound.ml.drift_detection.distribution import DistributionDriftConfig
from truthound.ml.drift_detection.feature import FeatureDriftConfig

# Distribution drift detection (use DistributionDriftConfig)
detector = DistributionDriftDetector(
    config=DistributionDriftConfig(method="psi", threshold=0.05)
)
detector.fit(baseline)
result = detector.detect(baseline, current)  # Both reference and current required

if result.is_drifted:
    print(f"Drift detected! Score: {result.drift_score}")
    # Get drifted columns from column_scores
    drifted = [col for col, score in result.column_scores if score >= 0.5]
    print(f"Drifted columns: {drifted}")

# Feature-level drift detection (use FeatureDriftConfig)
feature_detector = FeatureDriftDetector(
    config=FeatureDriftConfig(threshold=0.05)
)
feature_detector.fit(baseline)
result = feature_detector.detect(baseline, current)
```

### Rule Learning

핵심 개념과 경계에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml import PatternRuleLearner, DataProfileRuleLearner, ConstraintMiner
from truthound.ml.rule_learning.profile_learner import ProfileLearnerConfig
from truthound.ml.rule_learning.pattern_learner import PatternLearnerConfig
from truthound.ml.rule_learning.constraint_miner import ConstraintMinerConfig
import polars as pl

df = pl.DataFrame({
    "email": ["user@example.com", "admin@test.org", "info@company.io"],
    "age": [25, 30, 35, 40, 45],
    "status": ["active", "active", "inactive", "active", "pending"],
})

# Pattern-based rule learning (use PatternLearnerConfig)
pattern_learner = PatternRuleLearner(
    config=PatternLearnerConfig(min_pattern_ratio=0.9)
)
pattern_learner.fit(df.lazy())
result = pattern_learner.predict(df.lazy())
for rule in result.rules:
    print(f"Rule: {rule.name}, Column: {rule.column}")

# Profile-based rule learning (use ProfileLearnerConfig)
profile_learner = DataProfileRuleLearner(
    config=ProfileLearnerConfig(strictness="medium")
)
profile_learner.fit(df.lazy())
result = profile_learner.predict(df.lazy())

# Constraint mining (use ConstraintMinerConfig)
miner = ConstraintMiner(
    config=ConstraintMinerConfig(discover_functional_deps=True)
)
miner.fit(df.lazy())
result = miner.predict(df.lazy())
```

### Model Registry

핵심 개념과 경계에서 Manage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml import ModelRegistry, ZScoreAnomalyDetector, ModelState

# Create registry
registry = ModelRegistry()

# Register a model
detector = ZScoreAnomalyDetector()
registry.register(detector)

# List all models
models = registry.list_all()
print(f"Registered models: {models}")

# Model lifecycle: CREATED -> TRAINED -> DEPLOYED
detector.fit(df)  # State changes to TRAINED
print(f"Model state: {detector.state}")  # ModelState.TRAINED
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Lineage Module

핵심 개념과 경계에서 Lineage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Lineage Graph

핵심 개념과 경계에서 Build을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.lineage import LineageGraph, LineageNode, LineageEdge, NodeType, EdgeType

# Create a lineage graph
graph = LineageGraph()

# Add nodes
source = LineageNode(id="raw_data", name="Raw Data", node_type=NodeType.SOURCE)
transform = LineageNode(id="cleaned_data", name="Cleaned Data", node_type=NodeType.TRANSFORMATION)
target = LineageNode(id="analytics", name="Analytics Table", node_type=NodeType.TABLE)

graph.add_node(source)
graph.add_node(transform)
graph.add_node(target)

# Add edges (data flow) - no 'id' field required
graph.add_edge(LineageEdge(
    source="raw_data",
    target="cleaned_data",
    edge_type=EdgeType.TRANSFORMED_TO
))
graph.add_edge(LineageEdge(
    source="cleaned_data",
    target="analytics",
    edge_type=EdgeType.DERIVED_FROM
))

# Query lineage
upstream = graph.get_upstream("analytics")  # Returns list of LineageNode
downstream = graph.get_downstream("raw_data")
print(f"Upstream: {[n.id for n in upstream]}")
print(f"Downstream: {[n.id for n in downstream]}")

# Save and load graphs
graph.save("lineage.json")
loaded_graph = LineageGraph.load("lineage.json")

# Error handling for load()
# - Raises FileNotFoundError if file doesn't exist
# - Raises LineageError if file is empty or contains invalid JSON
```

#### Available NodeTypes

| 핵심 개념과 경계에서 NodeType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 핵심 개념과 경계에서 `SOURCE`, SOURCE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Raw data 소스 |
| 핵심 개념과 경계에서 `TABLE`, TABLE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 테이블 |
| 핵심 개념과 경계에서 `FILE`, FILE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일-based data |
| 핵심 개념과 경계에서 `STREAM`, STREAM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 소스 |
| 핵심 개념과 경계에서 `TRANSFORMATION`, TRANSFORMATION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `VALIDATION`, VALIDATION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 체크포인트 |
| 핵심 개념과 경계에서 `MODEL`, MODEL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `REPORT`, REPORT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 리포트 |
| 핵심 개념과 경계에서 `EXTERNAL`, EXTERNAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 External을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `VIRTUAL`, VIRTUAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Virtual/computed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

#### Available EdgeTypes

| 핵심 개념과 경계에서 EdgeType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 핵심 개념과 경계에서 `DERIVED_FROM`, DERIVED_FROM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `TRANSFORMED_TO`, TRANSFORMED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Transformation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `VALIDATED_BY`, VALIDATED_BY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 relationship |
| 핵심 개념과 경계에서 `USED_BY`, USED_BY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Usage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `JOINED_WITH`, JOINED_WITH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Join을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `AGGREGATED_TO`, AGGREGATED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Aggregation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `FILTERED_TO`, FILTERED_TO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `DEPENDS_ON`, DEPENDS_ON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Generic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Lineage Tracker

핵심 개념과 경계에서 Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.lineage import LineageTracker, OperationType
import polars as pl

# Create tracker
tracker = LineageTracker()

# Track a source
source_id = tracker.track_source(
    "sales_data.csv",
    source_type="file",  # file, table, stream, external
    location="/data/sales_data.csv",
    format="csv"
)

# Track transformations using dedicated methods
transform_id = tracker.track_transformation(
    "filtered_sales",
    sources=["sales_data.csv"],
    operation="filter",
)

# Track validation
validation_id = tracker.track_validation(
    "sales_validation",
    sources=["filtered_sales"],
    validators=["null", "range"],
)

# Track output
output_id = tracker.track_output(
    "sales_report",
    sources=["filtered_sales"],
    output_type="report",
)

# Get the lineage graph (use property, not method)
graph = tracker.graph
print(f"Nodes: {graph.node_count}, Edges: {graph.edge_count}")
```

#### 핵심 개념과 경계 개요

```python
with tracker.track("my_operation", OperationType.TRANSFORM) as ctx:
    # TrackingContext has sources and targets as lists
    # Use append() to add items (not add_source/add_target methods)
    ctx.sources.append("input_data")
    ctx.targets.append("output_data")
    # ... your transformation code ...
    print(f"Operation ID: {ctx.operation_id}")
    print(f"Sources: {ctx.sources}")
    print(f"Targets: {ctx.targets}")
```

### Impact Analysis

핵심 개념과 경계에서 Analyze을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.lineage import ImpactAnalyzer, LineageGraph

# Create analyzer with a lineage graph
graph = LineageGraph()
# ... add nodes and edges ...

analyzer = ImpactAnalyzer(graph)

# Analyze upstream impact (what feeds into this node?)
upstream_impact = analyzer.analyze_upstream("target_table")
print(f"Upstream nodes: {[n.id for n in upstream_impact.affected_nodes]}")

# Analyze downstream impact (what depends on this node?)
downstream_impact = analyzer.analyze_downstream("source_table")
print(f"Downstream nodes: {[n.id for n in downstream_impact.affected_nodes]}")

# Get impact severity
for node in downstream_impact.affected_nodes:
    print(f"Node: {node.id}, Severity: {node.impact_level}")
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Realtime Module

핵심 개념과 경계에서 Realtime을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Streaming Sources

핵심 개념과 경계에서 Connect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.realtime import MockStreamingSource, StreamingConfig

# Create a mock source for testing
source = MockStreamingSource(n_records=1000)

# Connect and read batches
with source as s:
    batch = s.read_batch(max_records=100)
    print(f"Read {len(batch)} records")
```

#### Available Sources

핵심 개념과 경계에서 Truthound을(를) 다루는 항목입니다:

핵심 개념과 경계에서 `IStreamAdapter`, Protocol-based, Adapters, IStreamAdapter을(를) 다루는 항목입니다:

| 어댑터 | 핵심 개념과 경계에서 Library을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설정 |
|---------|---------|---------------|
| 핵심 개념과 경계에서 `KafkaAdapter`, KafkaAdapter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `bootstrap_servers`, `topic`, `consumer_group`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `KinesisAdapter`, KinesisAdapter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `stream_name`, `region`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

핵심 개념과 경계에서 `StreamingSource`, StreamingSource, Pattern을(를) 다루는 항목입니다:

| 소스 | 핵심 개념과 경계에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설정 |
|--------|-------------|---------------|
| 핵심 개념과 경계에서 `MockStreamingSource`, MockStreamingSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Test을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `n_records`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `ParquetSource`, ParquetSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Parquet 파일 streaming | 핵심 개념과 경계에서 `path`, `batch_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `CSVSource`, CSVSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CSV 파일 streaming | 핵심 개념과 경계에서 `path`, `batch_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `PubSubSource`, PubSubSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Google, Cloud, Pub/Sub을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `project_id`, `subscription_id`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

핵심 개념과 경계에서 Note, Kafka, Kinesis, Pub/Sub, StreamingSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Incremental 검증

핵심 개념과 경계에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.realtime import IncrementalValidator, MemoryStateStore, StreamingConfig
import polars as pl

# Create state store
state_store = MemoryStateStore()

# Create incremental validator
validator = IncrementalValidator(
    state_store=state_store,
    track_duplicates=True,
    duplicate_columns=["id"],
)

# Validate batches
batch1 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
batch2 = pl.DataFrame({"id": [4, 5, 1], "value": [40, 50, 60]})  # id=1 is duplicate

result1 = validator.validate_batch(batch1)
print(f"Batch 1: {result1.record_count} records, {result1.issue_count} issues")

result2 = validator.validate_batch(batch2)
print(f"Batch 2: {result2.record_count} records, {result2.issue_count} issues")
```

### State Management

핵심 개념과 경계에서 Persist을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.realtime import MemoryStateStore

# Create in-memory state store
store = MemoryStateStore()

# Save state
store.set("validation_stats", {"total_records": 1000, "total_issues": 5})

# Load state
stats = store.get("validation_stats")
print(f"Total records: {stats['total_records']}")

# List all keys
keys = store.keys()

# Delete state
store.delete("validation_stats")

# Clear all state
store.clear()
```

### 체크포인트ing

핵심 개념과 경계에서 Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.realtime import CheckpointManager, MemoryStateStore
import tempfile

# Create checkpoint manager
with tempfile.TemporaryDirectory() as tmpdir:
    manager = CheckpointManager(checkpoint_dir=tmpdir)
    state_store = MemoryStateStore()

    # Create a checkpoint
    checkpoint = manager.create_checkpoint(
        state=state_store,
        batch_count=10,
        total_records=1000,
        total_issues=5,
    )
    print(f"Created checkpoint: {checkpoint.checkpoint_id}")

    # List checkpoints
    checkpoints = manager.list_checkpoints()

    # Get latest checkpoint (method is get_latest, not get_latest_checkpoint)
    latest = manager.get_latest()

    # Get checkpoint by ID
    restored = manager.get_checkpoint(checkpoint.checkpoint_id)
    print(f"Restored batch_count: {restored.batch_count}")

    # Restore state from checkpoint
    new_state = MemoryStateStore()
    manager.restore(checkpoint.checkpoint_id, new_state)
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CLI Commands

### ML 명령

```bash
# Anomaly detection
truthound ml anomaly data.csv --method zscore
truthound ml anomaly data.csv --method isolation_forest --contamination 0.1
truthound ml anomaly data.csv --method iqr --columns "amount,price"
truthound ml anomaly data.csv --method mad --output anomalies.json --format json

# Drift detection
truthound ml drift baseline.csv current.csv --method distribution --threshold 0.1
truthound ml drift train.parquet prod.parquet --method feature --threshold 0.2
truthound ml drift old.csv new.csv --method multivariate --output drift_report.json

# Rule learning
truthound ml learn-rules data.csv --output rules.json
truthound ml learn-rules data.csv --strictness strict --min-confidence 0.95
```

### Lineage Commands

```bash
# Show lineage graph
truthound lineage show lineage.json
truthound lineage show lineage.json --node my_table --direction upstream
truthound lineage show lineage.json --format dot > lineage.dot

# Impact analysis
truthound lineage impact lineage.json raw_data
truthound lineage impact lineage.json my_table --max-depth 3 --output impact.json

# Visualize lineage
truthound lineage visualize lineage.json -o graph.html
truthound lineage visualize lineage.json -o graph.html --renderer cytoscape --theme dark
truthound lineage visualize lineage.json -o graph.svg --renderer graphviz
truthound lineage visualize lineage.json -o graph.md --renderer mermaid
```

!!! note "참고"
    - 핵심 개념과 경계에서 HTML, `d3`, Interactive, D3.js을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - 핵심 개념과 경계에서 HTML, `cytoscape`, Cytoscape.js을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - 핵심 개념과 경계에서 `graphviz`, Static, Graphviz, SVG/PNG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - 핵심 개념과 경계에서 `mermaid`, Mermaid, Markdown을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 실시간 명령

!!! warning "참고"
핵심 개념과 경계에서 Truthound을(를) 다루는 항목입니다:

    - **`truthound checkpoint`**: For CI/CD 파이프라인 (YAML 설정 파일 based)
    - 핵심 개념과 경계에서 `truthound realtime checkpoint`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```bash
# Start streaming validation
truthound realtime validate mock --max-batches 10
truthound realtime validate mock --validators null,range --batch-size 500
truthound realtime validate kafka:my_topic --max-batches 100
truthound realtime validate kinesis:my_stream --batch-size 1000

# Monitor streaming metrics
truthound realtime monitor mock --interval 5 --duration 60
truthound realtime monitor kafka:my_topic --interval 10

# Manage streaming checkpoints (for fault tolerance in realtime validation)
# NOTE: This is different from `truthound checkpoint` which is for CI/CD!
truthound realtime checkpoint list --dir ./checkpoints
truthound realtime checkpoint show <checkpoint-id> --dir ./checkpoints
truthound realtime checkpoint delete <checkpoint-id> --dir ./checkpoints --force
```

!!! note "참고"
    - 핵심 개념과 경계에서 `mock`, Mock을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - 핵심 개념과 경계에서 `kafka:topic_name`, Kafka을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - 핵심 개념과 경계에서 `kinesis:stream_name`, Kinesis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 체크포인트 System Comparison

| 핵심 개념과 경계에서 Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Key, Options을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|---------|-------------|-------------|
| 핵심 개념과 경계에서 `truthound checkpoint run`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CI/CD 파이프라인 | 핵심 개념과 경계에서 `--config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Run 검증 based on YAML 설정 파일 |
| 핵심 개념과 경계에서 `truthound checkpoint list`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CI/CD 파이프라인 | 핵심 개념과 경계에서 `--config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | List 체크포인트 from 설정 파일 |
| 핵심 개념과 경계에서 `truthound checkpoint validate`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CI/CD 파이프라인 | 핵심 개념과 경계에서 `<config-file>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validate 설정 파일 |
| 핵심 개념과 경계에서 `truthound realtime checkpoint list`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 검증 | 핵심 개념과 경계에서 `--dir`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | List streaming state 체크포인트 |
| 핵심 개념과 경계에서 `truthound realtime checkpoint show`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 검증 | 핵심 개념과 경계에서 `--dir`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Show specific 체크포인트 details |
| 핵심 개념과 경계에서 `truthound realtime checkpoint delete`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming 검증 | 핵심 개념과 경계에서 `--dir`, `--force`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Delete 체크포인트 |

핵심 개념과 경계에서 CI/CD, Checkpoint, Example을(를) 다루는 항목입니다:
```bash
# Run checkpoint with YAML config file
truthound checkpoint run daily_data_validation --config truthound.yaml
truthound checkpoint list --config truthound.yaml
```

**Realtime 체크포인트 Example**:
```bash
# Manage streaming validation state
truthound realtime checkpoint list --dir ./streaming_checkpoints
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 레퍼런스

### ML Module

```python
from truthound.ml import (
    # Anomaly Detection
    AnomalyDetector,
    ZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
    IsolationForestDetector,
    EnsembleAnomalyDetector,

    # Drift Detection
    MLDriftDetector,
    DistributionDriftDetector,
    FeatureDriftDetector,

    # Rule Learning
    RuleLearner,
    PatternRuleLearner,
    DataProfileRuleLearner,
    ConstraintMiner,

    # Registry
    ModelRegistry,
    ModelType,
    ModelState,
)
```

### Lineage Module

```python
from truthound.lineage import (
    # Core
    LineageGraph,
    LineageNode,
    LineageEdge,

    # Tracking
    LineageTracker,

    # Analysis
    ImpactAnalyzer,

    # Enums
    NodeType,
    EdgeType,
    OperationType,
)
```

### Realtime Module

```python
from truthound.realtime import (
    # Protocol-based Adapters (async)
    # from truthound.realtime.adapters import KafkaAdapter, KinesisAdapter

    # StreamingSource Pattern (sync)
    StreamingSource,
    MockStreamingSource,
    PubSubSource,  # Uses StreamingSource pattern

    # Validation
    StreamingValidator,
    IncrementalValidator,
    BatchResult,

    # State
    StateStore,
    MemoryStateStore,

    # Checkpointing
    CheckpointManager,

    # Configuration
    StreamingConfig,
    StreamingMode,
    WindowConfig,
    WindowType,
)
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시

### Complete ML 파이프라인

```python
import polars as pl
from truthound.ml import (
    ZScoreAnomalyDetector,
    DistributionDriftDetector,
    DataProfileRuleLearner,
    ModelRegistry,
)

# Load data as LazyFrame
train_df = pl.scan_csv("train.csv")
prod_df = pl.scan_csv("production.csv")

# 1. Train anomaly detector
detector = ZScoreAnomalyDetector(threshold=3.0)
detector.fit(train_df)

# 2. Check for drift
drift_detector = DistributionDriftDetector(method="psi")
drift_detector.fit(train_df)
drift_result = drift_detector.detect(train_df, prod_df)

if drift_result.is_drifted:
    print("Warning: Data drift detected!")

# 3. Learn rules (can use DataFrame)
learner = DataProfileRuleLearner()
rules = learner.learn(train_df.collect())

# 4. Detect anomalies in production
anomaly_result = detector.predict(prod_df)
print(f"Found {anomaly_result.anomaly_count} anomalies")
```

### Complete Lineage Tracking

```python
from truthound.lineage import LineageTracker, ImpactAnalyzer, OperationType
import polars as pl

# Initialize tracker
tracker = LineageTracker()

# Track ETL pipeline
source_id = tracker.track_source("raw_sales.csv", source_type="file")

# Track cleaning transformation
clean_id = tracker.track_transformation(
    "cleaned_sales",
    sources=["raw_sales.csv"],
    operation="filter",
)

# Track aggregation
agg_id = tracker.track_transformation(
    "sales_by_region",
    sources=["cleaned_sales"],
    operation="aggregate",
)

# Analyze impact
graph = tracker.graph
analyzer = ImpactAnalyzer(graph)

impact = analyzer.analyze_downstream("raw_sales.csv")
print(f"Changing raw_sales.csv affects {len(impact.affected_nodes)} downstream nodes")
```

### Complete Streaming 검증

```python
from truthound.realtime import (
    MockStreamingSource,
    IncrementalValidator,
    MemoryStateStore,
    CheckpointManager,
)
import tempfile

# Setup
state_store = MemoryStateStore()
validator = IncrementalValidator(
    state_store=state_store,
    track_duplicates=True,
    duplicate_columns=["id"],
)

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmpdir)
    source = MockStreamingSource(n_records=1000)

    total_records = 0
    total_issues = 0
    batch_count = 0

    with source as s:
        while True:
            batch = s.read_batch(max_records=100)
            if len(batch) == 0:
                break

            result = validator.validate_batch(batch)
            total_records += result.record_count
            total_issues += result.issue_count
            batch_count += 1

            # Checkpoint every 5 batches
            if batch_count % 5 == 0:
                checkpoint_manager.create_checkpoint(
                    state=state_store,
                    batch_count=batch_count,
                    total_records=total_records,
                    total_issues=total_issues,
                )

    print(f"Processed {total_records} records with {total_issues} issues")
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 권장 방식

### ML Module

1. 핵심 개념과 경계에서 Always, Fit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 핵심 개념과 경계에서 Combine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 Monitor, Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Lineage Module

1. 핵심 개념과 경계에서 Track, Balance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 핵심 개념과 경계에서 IDs, Make을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 Regular을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 Document, Add을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Realtime Module

1. 핵심 개념과 경계에서 Choose, Balance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 핵심 개념과 경계에서 Checkpoint, Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 Monitor, Prevent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 Handle, Implement을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
