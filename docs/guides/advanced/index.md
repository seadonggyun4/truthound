# Advanced Features Guide

This guide covers enterprise-grade features with Truthound's Python API. It includes practical workflows for ML anomaly detection, data lineage tracking, real-time streaming validation, and plugin development.

---

## Quick Start

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

---

## Common Workflows

### Workflow 1: ML-Based Anomaly Detection Pipeline

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

### Workflow 2: Distribution Drift Detection

```python
import truthound as th

# Reference data (baseline)
baseline = "january_data.csv"
current = "february_data.csv"

# Quick drift detection with th.compare() - 14 methods available
# Statistical: auto, ks, psi, chi2, cvm, anderson
# Divergence: js, kl
# Distance: wasserstein, hellinger, bhattacharyya, tv, energy, mmd

# Auto-select method based on column type (recommended)
drift = th.compare(baseline, current, method="auto")

# Kolmogorov-Smirnov test (numeric columns)
drift = th.compare(baseline, current, method="ks")

# Population Stability Index (ML monitoring)
drift = th.compare(baseline, current, method="psi")

# Wasserstein distance (Earth Mover's Distance)
drift = th.compare(baseline, current, method="wasserstein")

# KL divergence (information theory)
drift = th.compare(baseline, current, method="kl")

# Cramér-von Mises (tail sensitivity)
drift = th.compare(baseline, current, method="cvm")

# Anderson-Darling (most sensitive to tails)
drift = th.compare(baseline, current, method="anderson")

# Hellinger distance (bounded metric)
drift = th.compare(baseline, current, method="hellinger")

# Bhattacharyya distance (classification bounds)
drift = th.compare(baseline, current, method="bhattacharyya")

# Total Variation distance (max probability diff)
drift = th.compare(baseline, current, method="tv")

# Energy distance (location/scale sensitivity)
drift = th.compare(baseline, current, method="energy")

# Maximum Mean Discrepancy (high-dimensional)
drift = th.compare(baseline, current, method="mmd")

# Check results
if drift.has_drift:
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"DRIFT: {col_drift.column} - {col_drift.result.method} = {col_drift.result.statistic:.4f}")
```

**Alternative: ML-based Drift Detection**

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

### Workflow 3: Data Lineage Tracking and Visualization

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

### Workflow 4: Real-Time Streaming Validation (Kafka)

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

### Workflow 5: Custom Plugin Development

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

---

## Full Documentation

This section covers Truthound's advanced features for enterprise-grade data quality validation.

---

## Overview

Truthound provides sophisticated capabilities beyond basic validation:

| Feature | Description | Use Case |
|---------|-------------|----------|
| [ML Anomaly Detection](ml-anomaly.md) | Statistical and ML-based anomaly/drift detection | Detect data quality issues automatically |
| [Data Lineage](lineage.md) | Track data transformations and dependencies | Impact analysis, compliance |
| [Plugin Architecture](plugins.md) | Extend Truthound with custom plugins | Custom validators, integrations |
| [Performance Tuning](performance.md) | Optimization strategies for large datasets | Enterprise scale processing |

---

## Quick Links

### ML Anomaly Detection

- **6 Anomaly Detectors**: Z-Score, IQR, MAD, Isolation Forest, Ensemble
- **14 Drift Methods**: auto, ks, psi, chi2, js, kl, wasserstein, cvm, anderson, hellinger, bhattacharyya, tv, energy, mmd
- **Model Monitoring**: Performance, drift, quality metrics with alerting

```python
import truthound as th

# Quick drift detection with th.compare()
drift = th.compare("baseline.csv", "current.csv", method="auto")      # Auto-select
drift = th.compare("baseline.csv", "current.csv", method="wasserstein")  # Earth Mover's
drift = th.compare("baseline.csv", "current.csv", method="anderson")     # Tail-sensitive
drift = th.compare("baseline.csv", "current.csv", method="hellinger")    # Bounded metric
drift = th.compare("baseline.csv", "current.csv", method="mmd")          # High-dimensional

# ML-based anomaly detection
from truthound import ml
detector = ml.IsolationForestDetector(contamination=0.1)
detector.fit(train_data)
result = detector.predict(test_data)
```

### Data Lineage

- **10 Node Types**: Source, Table, File, Stream, Transformation, Validation, Model, Report, External, Virtual
- **8 Edge Types**: Derived, Validated, Used, Transformed, Joined, Aggregated, Filtered, Depends
- **4 Visualization Renderers**: D3, Cytoscape, Graphviz, Mermaid
- **OpenLineage Integration**: Industry-standard lineage events

```python
from truthound.lineage import LineageTracker

tracker = LineageTracker()
tracker.track_source("raw_data", source_type="file")
tracker.track_transformation("cleaned", sources=["raw_data"])
tracker.track_validation("validated", sources=["cleaned"])
```

### Plugin Architecture

- **Security Sandbox**: NoOp, Process, Container isolation levels
- **Plugin Signing**: HMAC, RSA, Ed25519 algorithms
- **Version Constraints**: Semver support (^, ~, >=, <, ranges)
- **Hot Reload**: File watching with graceful reload and rollback

```python
from truthound.plugins import EnterprisePluginManager

manager = EnterprisePluginManager()
await manager.discover_plugins()
await manager.load_plugin("my-plugin")
```

### Performance Tuning

- **Expression Batching**: Multiple validators in single collect()
- **Lazy Loading**: 200+ validators loaded on demand
- **xxhash Caching**: ~10x faster fingerprinting
- **Enterprise Sampling**: Handle 100M+ row datasets

```python
from truthound.validators.base import ExpressionBatchExecutor

executor = ExpressionBatchExecutor()
executor.add_validator(NullValidator())
executor.add_validator(RangeValidator(min_value=0))
issues = executor.execute(lf)  # Single collect() for all
```

---

## Feature Matrix

| Feature | Open Source | Enterprise |
|---------|-------------|------------|
| Statistical Anomaly Detection | Yes | Yes |
| Isolation Forest | Yes | Yes |
| Ensemble Anomaly Detection | Yes | Yes |
| Distribution Drift (14 methods) | Yes | Yes |
| Statistical: KS, PSI, Chi2, CvM, Anderson | Yes | Yes |
| Divergence/Distance: JS, KL, Wasserstein, Hellinger, Bhattacharyya, TV, Energy, MMD | Yes | Yes |
| Concept Drift (DDM, ADWIN) | Yes | Yes |
| Model Monitoring | Yes | Yes |
| Data Lineage Tracking | Yes | Yes |
| Lineage Visualization | Yes | Yes |
| OpenLineage Integration | Yes | Yes |
| Plugin Security Sandbox | Yes | Yes |
| Plugin Signing/Verification | Yes | Yes |
| Hot Reload | Yes | Yes |
| Expression Batching | Yes | Yes |
| Enterprise Sampling | Yes | Yes |

---

## Next Steps

- [ML Anomaly Detection](ml-anomaly.md) - Learn about ML-based data quality monitoring
- [Data Lineage](lineage.md) - Track data transformations
- [Plugin Architecture](plugins.md) - Extend Truthound
- [Performance Tuning](performance.md) - Optimize for scale
