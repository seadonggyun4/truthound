# Advanced Features Guide

실무 운영 가이드에서 Truthound, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
# ML Anomaly Detection
from truthound.ml import IsolationForestDetector
detector = IsolationForestDetector(contamination=0.05)
detector.fit(training_data)
anomalies = detector.predict(new_data)

# Data Lineage
from truthound.lineage import LineageTracker
tracker = LineageTracker()
tracker.track_source("raw_data")
tracker.track_transformation("cleaned", sources=["raw_data"])
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 워크플로우s

### 워크플로우 1: ML-Based 이상치 Detection 파이프라인

```python
import polars as pl
from truthound.ml import IsolationForestDetector, EnsembleDetector, ZScoreDetector

# Load training data (known good data)
train_df = pl.read_csv("baseline_data.csv")

# Create ensemble detector (combines multiple algorithms)
detector = EnsembleDetector(
    detectors=[
        IsolationForestDetector(contamination=0.05),
        ZScoreDetector(threshold=3.0),
    ],
    strategy="voting",  # or "average"
)

# Train on baseline data
detector.fit(train_df)

# Detect anomalies in new data
new_df = pl.read_csv("new_data.csv")
result = detector.predict(new_df)

# Get anomaly scores and labels
print(f"Anomalies found: {result.anomaly_count}")
print(f"Anomaly ratio: {result.anomaly_ratio:.2%}")

# Get anomalous rows for investigation
anomalous_rows = new_df.filter(pl.Series(result.labels) == -1)
```

### 워크플로우 2: Distribution 드리프트 Detection

```python
from truthound.drift import compare

# Reference data (baseline)
baseline = "january_data.csv"
current = "february_data.csv"

# Quick drift detection with truthound.drift.compare() - 14 methods available
# Statistical: auto, ks, psi, chi2, cvm, anderson
# Divergence: js, kl
# Distance: wasserstein, hellinger, bhattacharyya, tv, energy, mmd

# Auto-select method based on column type (recommended)
drift = compare(baseline, current, method="auto")

# Kolmogorov-Smirnov test (numeric columns)
drift = compare(baseline, current, method="ks")

# Population Stability Index (ML monitoring)
drift = compare(baseline, current, method="psi")

# Wasserstein distance (Earth Mover's Distance)
drift = compare(baseline, current, method="wasserstein")

# KL divergence (information theory)
drift = compare(baseline, current, method="kl")

# Cramér-von Mises (tail sensitivity)
drift = compare(baseline, current, method="cvm")

# Anderson-Darling (most sensitive to tails)
drift = compare(baseline, current, method="anderson")

# Hellinger distance (bounded metric)
drift = compare(baseline, current, method="hellinger")

# Bhattacharyya distance (classification bounds)
drift = compare(baseline, current, method="bhattacharyya")

# Total Variation distance (max probability diff)
drift = compare(baseline, current, method="tv")

# Energy distance (location/scale sensitivity)
drift = compare(baseline, current, method="energy")

# Maximum Mean Discrepancy (high-dimensional)
drift = compare(baseline, current, method="mmd")

# Check results
if drift.has_drift:
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"DRIFT: {col_drift.column} - {col_drift.result.method} = {col_drift.result.statistic:.4f}")
```

**Alternative: ML-based 드리프트 Detection**

```python
from truthound.ml import DistributionDriftDetector
import polars as pl

reference_df = pl.read_csv("january_data.csv")
current_df = pl.read_csv("february_data.csv")

# Detect drift using PSI (Population Stability Index)
detector = DistributionDriftDetector(method="psi", threshold=0.1)
detector.fit(reference_df)
result = detector.detect(reference_df, current_df)

# Check for significant drift
for column, drift_info in result.column_drift.items():
    if drift_info.is_drifted:
        print(f"DRIFT: {column} - PSI: {drift_info.score:.3f}")
```

### 워크플로우 3: Data Lineage Tracking and Visualization

```python
from truthound.lineage import LineageTracker, LineageGraph
from truthound.lineage.visualization import MermaidRenderer

# Initialize tracker
tracker = LineageTracker()

# Track data flow
tracker.track_source("raw_orders", source_type="s3", path="s3://bucket/orders/")
tracker.track_source("raw_customers", source_type="postgres", table="customers")

tracker.track_transformation(
    "cleaned_orders",
    sources=["raw_orders"],
    operation="filter_nulls",
)

tracker.track_transformation(
    "enriched_orders",
    sources=["cleaned_orders", "raw_customers"],
    operation="join",
)

tracker.track_validation(
    "validated_orders",
    sources=["enriched_orders"],
    checkpoint="order_quality",
)

# Visualize lineage
graph = tracker.build_graph()
renderer = MermaidRenderer()
mermaid_code = renderer.render(graph)
print(mermaid_code)

# Save as HTML
from truthound.lineage.visualization import D3Renderer
d3_renderer = D3Renderer()
html = d3_renderer.render(graph, output_path="lineage.html")
```

### 워크플로우 4: Real-Time Streaming 검증 (Kafka)

```python
import asyncio
from truthound.realtime.adapters import KafkaAdapter
from truthound.realtime import StreamingValidator
import truthound as th

async def validate_stream():
    # Configure Kafka adapter
    adapter = KafkaAdapter(
        bootstrap_servers="localhost:9092",
        topic="user-events",
        group_id="truthound-validator",
    )

    # Create streaming validator
    validator = StreamingValidator(
        adapter=adapter,
        validators=["null", "email", "range"],
        batch_size=1000,
        batch_timeout_ms=5000,
    )

    # Process stream
    async for batch_result in validator.validate_stream():
        if batch_result.issues:
            print(f"Batch {batch_result.batch_id}: {len(batch_result.issues)} issues")
            # Send alerts, store results, etc.

asyncio.run(validate_stream())
```

### 워크플로우 5: Custom Plugin Development

```python
from truthound.plugins import PluginBase, PluginConfig, plugin_registry
from truthound.validators import Validator

class MyCustomPlugin(PluginBase):
    name = "my-custom-plugin"
    version = "1.0.0"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.9)

    def on_load(self):
        # Register custom validator
        @plugin_registry.register_validator
        class CustomValidator(Validator):
            name = "custom_check"

            def validate(self, lf):
                # Custom validation logic
                pass

    def on_unload(self):
        # Cleanup
        pass

# Load plugin
from truthound.plugins import EnterprisePluginManager

manager = EnterprisePluginManager()
await manager.load_plugin("my-custom-plugin", config={"threshold": 0.95})
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Full Documentation

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|----------|
| [ML 이상치 Detection](ml-anomaly.md) | 실무 운영 가이드에서 Statistical, ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Detect 데이터 품질 issues automatically |
| 실무 운영 가이드에서 Data, Lineage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Impact을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| [Plugin 아키텍처](plugins.md) | 실무 운영 가이드에서 Truthound, Extend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Custom 검증기, integrations |
| [성능 Tuning](performance.md) | 실무 운영 가이드에서 Optimization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enterprise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Quick Links

### ML 이상치 Detection

- 실무 운영 가이드에서 Anomaly, Detectors, Z-Score, IQR, MAD, Isolation, Forest, Ensemble을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Drift, Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Model, Monitoring, Performance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.drift import compare

# Quick drift detection with truthound.drift.compare()
drift = compare("baseline.csv", "current.csv", method="auto")         # Auto-select
drift = compare("baseline.csv", "current.csv", method="wasserstein")  # Earth Mover's
drift = compare("baseline.csv", "current.csv", method="anderson")     # Tail-sensitive
drift = compare("baseline.csv", "current.csv", method="hellinger")    # Bounded metric
drift = compare("baseline.csv", "current.csv", method="mmd")          # High-dimensional

# ML-based anomaly detection
from truthound import ml
detector = ml.IsolationForestDetector(contamination=0.1)
detector.fit(train_data)
result = detector.predict(test_data)
```

### Data Lineage

- 실무 운영 가이드에서 Node, Types, Source, Table, File, Stream, Transformation, Validation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Edge, Types, Derived, Validated, Used, Transformed, Joined, Aggregated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Visualization, Renderers, Cytoscape, Graphviz, Mermaid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 OpenLineage, Integration, Industry-standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.lineage import LineageTracker

tracker = LineageTracker()
tracker.track_source("raw_data", source_type="file")
tracker.track_transformation("cleaned", sources=["raw_data"])
tracker.track_validation("validated", sources=["cleaned"])
```

### Plugin 아키텍처

- 실무 운영 가이드에서 Security, Sandbox, NoOp, Process, Container을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Plugin, Signing, HMAC, RSA, Ed25519을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Version, Constraints, Semver을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Hot, Reload, File을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.plugins import EnterprisePluginManager

manager = EnterprisePluginManager()
await manager.discover_plugins()
await manager.load_plugin("my-plugin")
```

### 성능 Tuning

- 실무 운영 가이드에서 Expression, Batching, Multiple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Lazy, Loading을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- **xxhash 캐싱**: ~10x faster fingerprinting
- 실무 운영 가이드에서 Enterprise, Sampling, Handle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.validators.base import ExpressionBatchExecutor

executor = ExpressionBatchExecutor()
executor.add_validator(NullValidator())
executor.add_validator(RangeValidator(min_value=0))
issues = executor.execute(lf)  # Single collect() for all
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Feature Matrix

| 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Open 소스 | 실무 운영 가이드에서 Enterprise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|------------|
| Statistical 이상치 Detection | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Ensemble 이상치 Detection | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Distribution 드리프트 (14 methods) | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Statistical, PSI, Chi2, CvM, Anderson을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Divergence/Distance, Wasserstein, Hellinger, Bhattacharyya, Energy, MMD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Concept 드리프트 (DDM, ADWIN) | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Model 모니터링 | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Data, Lineage, Tracking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Lineage, Visualization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| OpenLineage 통합 | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Plugin 보안 Sandbox | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Plugin, Signing/Verification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Hot, Reload을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Expression, Batching을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Enterprise, Sampling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 실무 운영 가이드에서 Anomaly, Detection, ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Data, Lineage, Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Truthound, Plugin, Architecture, Extend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Performance, Tuning, Optimize을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
