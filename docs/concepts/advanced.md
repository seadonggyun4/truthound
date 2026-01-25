# Advanced Features (Phase 10)

This document describes the advanced features introduced in Phase 10 of Truthound: **ML Module**, **Lineage Module**, and **Realtime Module**.

---

## Table of Contents

1. [Overview](#overview)
2. [ML Module](#ml-module)
   - [Anomaly Detection](#anomaly-detection)
   - [Drift Detection](#drift-detection)
   - [Rule Learning](#rule-learning)
   - [Model Registry](#model-registry)
3. [Lineage Module](#lineage-module)
   - [Lineage Graph](#lineage-graph)
   - [Lineage Tracker](#lineage-tracker)
   - [Impact Analysis](#impact-analysis)
4. [Realtime Module](#realtime-module)
   - [Streaming Sources](#streaming-sources)
   - [Incremental Validation](#incremental-validation)
   - [State Management](#state-management)
   - [Checkpointing](#checkpointing)
5. [CLI Commands](#cli-commands)
6. [API Reference](#api-reference)

---

## Overview

Phase 10 introduces three major modules that extend Truthound's capabilities into advanced data quality scenarios:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **ML** | Machine learning-based data quality | Anomaly detection, drift detection, rule learning |
| **Lineage** | Data flow tracking | Graph-based lineage, impact analysis |
| **Realtime** | Streaming validation | Incremental validation, checkpointing |

---

## ML Module

The ML module provides machine learning capabilities for data quality validation.

### Anomaly Detection

Detect anomalies in your data using various statistical and ML-based methods.

> **Note**: All ML detectors require `LazyFrame` input and at least 10 samples for training.

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

| Detector | Method | Best For |
|----------|--------|----------|
| `ZScoreAnomalyDetector` | Z-Score | Normally distributed data |
| `IQRAnomalyDetector` | Interquartile Range | Skewed distributions |
| `MADAnomalyDetector` | Median Absolute Deviation | Robust to outliers |
| `IsolationForestDetector` | Isolation Forest | High-dimensional data |
| `EnsembleAnomalyDetector` | Voting ensemble | Combining multiple detectors |

### Drift Detection

Detect distribution drift between baseline and current datasets.

> **Note**: Drift detectors require `LazyFrame` input and at least 10 samples.

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

Automatically learn validation rules from your data.

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

Manage ML model lifecycle.

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

---

## Lineage Module

The Lineage module provides data lineage tracking and impact analysis.

### Lineage Graph

Build and query data flow graphs.

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

| NodeType | Description |
|----------|-------------|
| `SOURCE` | Raw data source |
| `TABLE` | Database table |
| `FILE` | File-based data |
| `STREAM` | Streaming source |
| `TRANSFORMATION` | Data transformation |
| `VALIDATION` | Validation checkpoint |
| `MODEL` | ML model |
| `REPORT` | Output report |
| `EXTERNAL` | External system |
| `VIRTUAL` | Virtual/computed dataset |

#### Available EdgeTypes

| EdgeType | Description |
|----------|-------------|
| `DERIVED_FROM` | Data derivation |
| `TRANSFORMED_TO` | Transformation |
| `VALIDATED_BY` | Validation relationship |
| `USED_BY` | Usage relationship |
| `JOINED_WITH` | Join operation |
| `AGGREGATED_TO` | Aggregation |
| `FILTERED_TO` | Filter operation |
| `DEPENDS_ON` | Generic dependency |

### Lineage Tracker

Track data operations with dedicated methods for each operation type.

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

#### Alternative: Context Manager for Complex Operations

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

Analyze the impact of changes in your data pipeline.

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

---

## Realtime Module

The Realtime module provides streaming data validation capabilities.

### Streaming Sources

Connect to various streaming sources.

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

Truthound provides two different streaming architectures:

**Protocol-based Adapters** (async, implements `IStreamAdapter`):

| Adapter | Library | Configuration |
|---------|---------|---------------|
| `KafkaAdapter` | aiokafka | `bootstrap_servers`, `topic`, `consumer_group` |
| `KinesisAdapter` | aiobotocore | `stream_name`, `region` |

**StreamingSource Pattern** (sync, extends `StreamingSource` base class):

| Source | Description | Configuration |
|--------|-------------|---------------|
| `MockStreamingSource` | Test source with generated data | `n_records` |
| `ParquetSource` | Parquet file streaming | `path`, `batch_size` |
| `CSVSource` | CSV file streaming | `path`, `batch_size` |
| `PubSubSource` | Google Cloud Pub/Sub | `project_id`, `subscription_id` |

Note: Kafka and Kinesis use the protocol-based adapter architecture with async operations. Pub/Sub uses the StreamingSource pattern with synchronous operations.

### Incremental Validation

Validate data incrementally with state management.

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

Persist validation state across batches.

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

### Checkpointing

Create and restore checkpoints for fault tolerance.

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

---

## CLI Commands

### ML Commands

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

!!! note "Supported Renderers"
    - `d3`: Interactive D3.js (HTML)
    - `cytoscape`: Cytoscape.js (HTML)
    - `graphviz`: Static Graphviz (SVG/PNG)
    - `mermaid`: Mermaid diagram (Markdown)

### Realtime Commands

!!! warning "checkpoint vs realtime checkpoint"
    Truthound has two different checkpoint systems:

    - **`truthound checkpoint`**: For CI/CD pipelines (YAML configuration file based)
    - **`truthound realtime checkpoint`**: For streaming validation (directory-based state management)

    These two commands serve completely different purposes and have different options.

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

!!! note "Streaming Source Formats"
    - `mock`: Mock data source for testing
    - `kafka:topic_name`: Kafka topic (requires aiokafka)
    - `kinesis:stream_name`: Kinesis stream (requires aiobotocore)

### Checkpoint System Comparison

| Command | Purpose | Key Options | Description |
|---------|---------|-------------|-------------|
| `truthound checkpoint run` | CI/CD pipeline | `--config` | Run validation based on YAML config file |
| `truthound checkpoint list` | CI/CD pipeline | `--config` | List checkpoints from config file |
| `truthound checkpoint validate` | CI/CD pipeline | `<config-file>` | Validate config file |
| `truthound realtime checkpoint list` | Streaming validation | `--dir` | List streaming state checkpoints |
| `truthound realtime checkpoint show` | Streaming validation | `--dir` | Show specific checkpoint details |
| `truthound realtime checkpoint delete` | Streaming validation | `--dir`, `--force` | Delete checkpoint |

**CI/CD Checkpoint Example** (see [docs/guides/ci-cd.md](../guides/ci-cd.md)):
```bash
# Run checkpoint with YAML config file
truthound checkpoint run daily_data_validation --config truthound.yaml
truthound checkpoint list --config truthound.yaml
```

**Realtime Checkpoint Example**:
```bash
# Manage streaming validation state
truthound realtime checkpoint list --dir ./streaming_checkpoints
```

---

## API Reference

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

---

## Examples

### Complete ML Pipeline

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

### Complete Streaming Validation

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

---

## Best Practices

### ML Module

1. **Always fit on training data**: Fit detectors on clean, representative data
2. **Use ensemble for robustness**: Combine multiple detectors for better accuracy
3. **Monitor drift continuously**: Set up regular drift checks in production
4. **Version your models**: Use the registry to track model versions

### Lineage Module

1. **Track at appropriate granularity**: Balance detail vs. overhead
2. **Use meaningful node IDs**: Make lineage graphs readable
3. **Regular impact analysis**: Before major changes, check impact
4. **Document transformations**: Add metadata to operations

### Realtime Module

1. **Choose appropriate batch sizes**: Balance latency vs. throughput
2. **Checkpoint frequently**: Enable recovery from failures
3. **Monitor state store size**: Prevent memory issues with large state
4. **Handle backpressure**: Implement proper flow control
