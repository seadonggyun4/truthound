# Advanced Features

Advanced Python API for ML, Lineage, Real-time streaming, and Profiling.

## Parallel Execution

For large datasets with many validators, use parallel execution for better performance.

### Basic Parallel Execution

```python
import truthound as th

# Enable parallel execution (uses all available cores)
report = th.check("large_data.csv", parallel=True)

# Control thread count
report = th.check("large_data.csv", parallel=True, max_workers=4)
```

### DAG-Based Execution

Validators are organized into a Directed Acyclic Graph (DAG) based on dependencies:

```python
from truthound.validators.optimization.orchestrator import (
    ValidatorDAG,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
)

# Build DAG from validators
dag = ValidatorDAG()
dag.add_validators(validator_instances)
plan = dag.build_execution_plan()

# Execute with adaptive strategy (auto-selects parallelism)
strategy = AdaptiveExecutionStrategy()
result = plan.execute(lf, strategy)
```

---

## Query Pushdown

For SQL data sources, push validation logic to the database server:

```python
import truthound as th
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="large_table",
    host="localhost",
    database="mydb",
    user="postgres",
)

# Enable pushdown - validations execute server-side
report = th.check(source=source, pushdown=True)

# Example: null_check becomes:
# SELECT COUNT(*) FROM table WHERE column IS NULL
```

### Pushdown Benefits

| Benefit | Description |
|---------|-------------|
| Reduced data transfer | Only aggregated results are returned |
| Database optimization | Leverages database query optimizer |
| Scalability | Handles billions of rows efficiently |

---

## ML Module

Machine learning-based validation and anomaly detection.

### Anomaly Detection

```python
from truthound import ml
import polars as pl

# Statistical anomaly detectors
detector = ml.ZScoreAnomalyDetector(threshold=3.0)
detector = ml.IQRAnomalyDetector(multiplier=1.5)
detector = ml.MADAnomalyDetector(threshold=3.5)

# Isolation Forest (requires scikit-learn)
detector = ml.IsolationForestDetector(
    n_estimators=100,
    contamination=0.01,
)

# Ensemble detector (combines multiple methods)
detector = ml.EnsembleAnomalyDetector(
    detectors=[
        ml.ZScoreAnomalyDetector(),
        ml.IQRAnomalyDetector(),
    ],
    voting="majority",  # or "any", "all"
)

# Fit and predict
df = pl.read_csv("data.csv")
detector.fit(df, columns=["amount", "count"])
result = detector.predict(df)

print(f"Anomalies: {result.anomaly_count}")
for score in result.scores:
    if score.is_anomaly:
        print(f"  Row {score.row_index}: {score.anomaly_type}")
```

### ML Drift Detection

```python
from truthound import ml

# Distribution drift detector
detector = ml.DistributionDriftDetector(
    method="ks",  # "ks", "psi", "chi2", "js"
    threshold=0.05,
)

# Feature drift detector (multi-column)
detector = ml.FeatureDriftDetector(
    columns=["age", "income", "score"],
    method="psi",
)

# Fit baseline
detector.fit(baseline_df)

# Detect drift
result = detector.detect(current_df)
if result.has_drift:
    print(f"Drift detected: {result.drifted_columns}")
```

### Rule Learning

```python
from truthound import ml

# Learn rules from data
learner = ml.DataProfileRuleLearner()
result = learner.learn(df)

for rule in result.rules:
    print(f"{rule.column}: {rule.rule_type} - {rule.parameters}")

# Pattern-based rule learning
learner = ml.PatternRuleLearner()
result = learner.learn(df, columns=["email", "phone"])

# Constraint mining
miner = ml.ConstraintMiner()
constraints = miner.mine(df)
```

### Model Registry

```python
from truthound import ml

# Global registry
registry = ml.model_registry

# Register custom model
@ml.register_model("my_detector")
class MyDetector(ml.AnomalyDetector):
    def fit(self, data, columns):
        ...
    def predict(self, data):
        ...

# Use registered model
detector = registry.create("my_detector")
```

### ML Module Classes

| Category | Classes |
|----------|---------|
| **Anomaly Detectors** | `IsolationForestDetector`, `StatisticalAnomalyDetector`, `ZScoreAnomalyDetector`, `IQRAnomalyDetector`, `MADAnomalyDetector`, `EnsembleAnomalyDetector` |
| **Drift Detectors** | `DistributionDriftDetector`, `FeatureDriftDetector`, `ConceptDriftDetector`, `MultivariateDriftDetector` |
| **Rule Learners** | `DataProfileRuleLearner`, `ConstraintMiner`, `PatternRuleLearner` |
| **Base Classes** | `MLModel`, `AnomalyDetector`, `MLDriftDetector`, `RuleLearner` |
| **Results** | `AnomalyScore`, `AnomalyResult`, `DriftResult`, `LearnedRule`, `RuleLearningResult`, `ModelInfo` |
| **Configurations** | `MLConfig`, `AnomalyConfig`, `DriftConfig`, `RuleLearningConfig` |
| **Enums** | `ModelType`, `ModelState`, `AnomalyType`, `SeverityLevel` |
| **Registry** | `ModelRegistry`, `model_registry`, `register_model` |
| **Exceptions** | `MLError`, `ModelNotTrainedError`, `ModelTrainingError`, `ModelLoadError`, `InsufficientDataError` |

---

## Lineage Module

Track data flow and analyze impact of changes.

### Basic Lineage Tracking

```python
from truthound import lineage

# Create tracker
tracker = lineage.LineageTracker()

# Track data sources
tracker.track_source(
    "raw_customers",
    source_type="csv",
    path="/data/customers.csv",
)

tracker.track_source(
    "raw_orders",
    source_type="database",
    connection="postgresql://localhost/db",
    table="orders",
)

# Track transformations
tracker.track_transformation(
    "cleaned_customers",
    sources=["raw_customers"],
    operation="clean",
    metadata={"removed_nulls": True},
)

tracker.track_transformation(
    "customer_orders",
    sources=["cleaned_customers", "raw_orders"],
    operation="join",
)

# Track validation
tracker.track_validation(
    "validated_data",
    sources=["customer_orders"],
    validators=["null", "range", "format"],
)
```

### Impact Analysis

```python
from truthound import lineage

analyzer = lineage.ImpactAnalyzer(tracker.get_graph())

# Forward impact: what depends on this node?
impact = analyzer.analyze_impact("raw_customers")
print(f"Affected nodes: {[n.node_id for n in impact.affected_nodes]}")

# Backward lineage: where does this data come from?
sources = analyzer.trace_sources("validated_data")
print(f"Source nodes: {sources}")
```

### Visualization

```python
from truthound.lineage.visualization import (
    D3Renderer,
    CytoscapeRenderer,
    GraphvizRenderer,
    MermaidRenderer,
    RenderConfig,
)

graph = tracker.get_graph()

# D3.js visualization (interactive HTML)
renderer = D3Renderer()
html = renderer.render_html(graph, RenderConfig(
    layout="dagre",
    node_color_by="type",
))
with open("lineage.html", "w") as f:
    f.write(html)

# Mermaid diagram (for documentation)
renderer = MermaidRenderer()
mermaid_code = renderer.render(graph)
print(mermaid_code)

# Graphviz (for static images)
renderer = GraphvizRenderer()
dot = renderer.render(graph)
```

### OpenLineage Integration

```python
from truthound.lineage.integrations.openlineage import OpenLineageEmitter

emitter = OpenLineageEmitter(
    api_url="http://localhost:5000",
    namespace="my-pipeline",
)

# Start a run
run = emitter.start_run(
    job_name="data-validation",
    job_namespace="truthound",
)

# Emit completion with outputs
emitter.emit_complete(
    run,
    outputs=[
        {"namespace": "file", "name": "/data/validated.parquet"},
    ],
)
```

### Lineage Module Classes

| Category | Classes |
|----------|---------|
| **Core** | `LineageTracker`, `TrackingContext`, `LineageGraph`, `LineageNode`, `LineageEdge` |
| **Analysis** | `ImpactAnalyzer`, `ImpactResult`, `AffectedNode`, `ImpactLevel` |
| **Visualization** | `D3Renderer`, `CytoscapeRenderer`, `GraphvizRenderer`, `MermaidRenderer`, `RenderConfig` |
| **Enums** | `NodeType`, `EdgeType`, `OperationType` |
| **Data Classes** | `LineageMetadata`, `LineageConfig` |
| **Exceptions** | `LineageError`, `NodeNotFoundError`, `CyclicDependencyError` |
| **Integration** | `OpenLineageEmitter` |

---

## Real-time Module

Streaming and incremental validation.

### Streaming Validator

```python
from truthound import realtime

# Create streaming validator
validator = realtime.StreamingValidator(
    validators=["null", "range", "format"],
    window_size=1000,
    mode=realtime.StreamingMode.MICRO_BATCH,
)

# Process batches
async for batch in data_stream:
    result = validator.validate_batch(batch)
    if result.has_issues:
        for issue in result.issues:
            print(f"Issue: {issue.column} - {issue.issue_type}")
```

### Incremental Validator

```python
from truthound import realtime

# Create incremental validator with state
validator = realtime.IncrementalValidator(
    validators=["unique", "aggregate"],
    state_store=realtime.MemoryStateStore(),
)

# Process increments
for chunk in data_chunks:
    result = validator.validate_increment(chunk)
    print(f"Total rows processed: {validator.total_rows}")
```

### Kafka Integration

```python
from truthound.realtime.adapters import KafkaAdapter, KafkaAdapterConfig
from truthound.realtime.factory import StreamAdapterFactory

# Create via factory
adapter = StreamAdapterFactory.create("kafka", {
    "bootstrap_servers": "localhost:9092",
    "topic": "events",
    "group_id": "truthound-validators",
})

# Or direct instantiation
adapter = KafkaAdapter(KafkaAdapterConfig(
    bootstrap_servers="localhost:9092",
    topic="events",
    group_id="truthound-validators",
    auto_offset_reset="earliest",
))

async with adapter:
    async for message in adapter.consume():
        # Validate message
        df = message.to_polars()
        report = th.check(df, validators=["null", "range"])

        if not report.has_issues:
            await adapter.commit(message)
```

### Kinesis Integration

```python
from truthound.realtime.adapters import KinesisAdapter, KinesisAdapterConfig

adapter = KinesisAdapter(KinesisAdapterConfig(
    stream_name="my-stream",
    region_name="us-east-1",
    shard_iterator_type="LATEST",
))

async with adapter:
    async for message in adapter.consume():
        process_message(message)
```

### Window-Based Validation

```python
from truthound import realtime

validator = realtime.StreamingValidator(
    validators=["null", "range"],
    window_config=realtime.WindowConfig(
        type=realtime.WindowType.TUMBLING,
        size_seconds=60,
    ),
)

# Results are aggregated per window
async for window_result in validator.validate_stream(stream):
    print(f"Window {window_result.window_id}:")
    print(f"  Rows: {window_result.row_count}")
    print(f"  Issues: {len(window_result.issues)}")
```

### Realtime Module Classes

| Category | Classes |
|----------|---------|
| **Validators** | `StreamingValidator`, `IncrementalValidator` |
| **Sources (Legacy)** | `StreamingSource`, `KafkaSource`, `KinesisSource`, `PubSubSource`, `MockStreamingSource` |
| **State** | `StateStore`, `MemoryStateStore`, `CheckpointManager` |
| **Protocols** | `IStreamSource`, `IStreamSink`, `IStreamProcessor`, `IStateStoreProtocol`, `IMetricsCollector` |
| **Data Classes** | `StreamMessage`, `MessageBatch`, `MessageHeader`, `StreamMetrics` |
| **Results** | `BatchResult`, `WindowResult` |
| **Configs** | `StreamingConfig`, `WindowConfig` |
| **Enums** | `StreamingMode`, `WindowType`, `TriggerType`, `DeserializationFormat`, `OffsetReset`, `AckMode` |
| **Factory** | `StreamAdapterFactory` |
| **Exceptions** | `StreamingError`, `ConnectionError`, `TimeoutError` |

---

## Profiler Module

Advanced data profiling and rule generation.

### Data Profiler

```python
from truthound import profiler

# Profile a file
profile = profiler.profile_file("data.parquet")

print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
print(f"Memory: {profile.estimated_memory_bytes / 1024 / 1024:.1f} MB")

# Profile columns
for col in profile:
    print(f"\n{col.name} ({col.inferred_type}):")
    print(f"  Null ratio: {col.null_ratio:.2%}")
    print(f"  Unique ratio: {col.unique_ratio:.2%}")
    if col.numeric_stats:
        print(f"  Mean: {col.numeric_stats.mean:.2f}")
        print(f"  Std: {col.numeric_stats.std:.2f}")
```

### Profile DataFrame

```python
from truthound import profiler
import polars as pl

df = pl.read_csv("data.csv")
profile = profiler.profile_dataframe(df, name="my_data")

# Save profile
profiler.save_profile(profile, "profile.json")

# Load profile
loaded = profiler.load_profile("profile.json")
```

### Validation Suite Generation

```python
from truthound import profiler

# Generate validation suite from profile
suite = profiler.generate_suite(
    profile,
    strictness="medium",  # "low", "medium", "high"
    include_categories=["schema", "completeness", "format"],
)

# Export as YAML
yaml_rules = suite.to_yaml()
with open("validation_rules.yaml", "w") as f:
    f.write(yaml_rules)

# Export as Python code
python_code = suite.to_python_code()
print(python_code)

# Export as JSON
json_rules = suite.to_json()
```

### Custom Profiler

```python
from truthound import profiler

# Create profiler with custom config
prof = profiler.DataProfiler(
    config=profiler.ProfilerConfig(
        sample_size=10000,
        pattern_detection=True,
        correlation_analysis=True,
        histogram_bins=50,
    )
)

# Profile with custom analyzers
profile = prof.profile(
    df,
    analyzers=[
        profiler.BasicStatsAnalyzer(),
        profiler.PatternAnalyzer(),
        profiler.CorrelationAnalyzer(),
    ],
)
```

---

## Data Docs Module

Generate HTML reports and documentation.

### Basic HTML Report

```python
from truthound import datadocs

# Generate HTML report from validation result
html = datadocs.generate_html_report(report)
with open("report.html", "w") as f:
    f.write(html)

# From file
datadocs.generate_report_from_file(
    "validation_result.json",
    output_path="report.html",
)
```

### Report Builder

```python
from truthound import datadocs

builder = datadocs.HTMLReportBuilder(
    config=datadocs.ReportConfig(
        title="Data Quality Report",
        include_charts=True,
        include_samples=True,
        max_issues=100,
    ),
    theme=datadocs.ReportTheme.DARK,
    chart_library=datadocs.ChartLibrary.PLOTLY,
)

html = builder.build(report)
```

### Custom Themes

```python
from truthound import datadocs
from truthound.datadocs.themes import ThemeConfig

# Use built-in theme
builder = datadocs.HTMLReportBuilder(
    theme=datadocs.ReportTheme.PROFESSIONAL,
)

# Custom theme
custom_theme = ThemeConfig(
    primary_color="#1a73e8",
    secondary_color="#34a853",
    background_color="#ffffff",
    font_family="Inter, sans-serif",
    logo_url="https://example.com/logo.png",
)
builder = datadocs.HTMLReportBuilder(theme=custom_theme)
```

---

## Checkpoint Module

CI/CD integration and validation pipelines.

### Basic Checkpoint

```python
from truthound import checkpoint

# Create checkpoint runner
runner = checkpoint.CheckpointRunner(
    config=checkpoint.CheckpointConfig(
        fail_on_critical=True,
        fail_on_high=False,
        notify_on_failure=True,
    )
)

# Run validation
result = runner.run(
    data="data.csv",
    validators=["null", "duplicate", "range"],
)

# Check result
if result.passed:
    print("Validation passed!")
else:
    print(f"Failed: {len(result.issues)} issues")
    runner.notify(result)  # Send notifications
```

### Notification Providers

```python
from truthound.checkpoint.actions import (
    SlackNotifier,
    EmailNotifier,
    PagerDutyNotifier,
    WebhookNotifier,
)

# Slack
notifier = SlackNotifier(
    webhook_url="https://hooks.slack.com/...",
    channel="#data-quality",
)

# Email
notifier = EmailNotifier(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@example.com",
    recipients=["team@example.com"],
)

# PagerDuty
notifier = PagerDutyNotifier(
    routing_key="your-routing-key",
    severity_mapping={"critical": "critical", "high": "error"},
)
```

---

## Type Definitions

### Import Metrics

```python
from truthound import get_truthound_import_metrics

# Get lazy loading metrics
metrics = get_truthound_import_metrics()
print(f"Total lazy loads: {metrics['total_lazy_loads']}")
print(f"Total load time: {metrics['total_load_time_ms']:.2f}ms")
print(f"Slowest loads: {metrics['slowest_loads'][:5]}")
```

### Module Structure

| Module | Description |
|--------|-------------|
| `truthound.ml` | ML anomaly/drift detection, rule learning |
| `truthound.lineage` | Data lineage tracking and visualization |
| `truthound.realtime` | Streaming and incremental validation |
| `truthound.profiler` | Data profiling and rule generation |
| `truthound.datadocs` | HTML report generation |
| `truthound.checkpoint` | CI/CD integration |

## See Also

- [Core Functions](core-functions.md) - Basic API functions
- [Validators](validators.md) - Validator interface
- [Data Sources](datasources.md) - Database connections
- [Reporters](reporters.md) - Output formatters
