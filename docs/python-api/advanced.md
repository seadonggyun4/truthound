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
from truthound.ml.anomaly_models.statistical import StatisticalConfig
from truthound.ml.anomaly_models.isolation_forest import IsolationForestConfig
from truthound.ml.anomaly_models.ensemble import EnsembleConfig, EnsembleStrategy
import polars as pl

# Statistical anomaly detectors (using Config objects)
detector = ml.ZScoreAnomalyDetector(
    config=StatisticalConfig(
        z_threshold=3.0,
        columns=["amount", "count"],  # specify columns in config
    )
)
detector = ml.IQRAnomalyDetector(
    config=StatisticalConfig(
        iqr_multiplier=1.5,
    )
)
detector = ml.MADAnomalyDetector(
    config=StatisticalConfig(
        z_threshold=3.5,  # MAD uses z_threshold for scaling
    )
)

# Isolation Forest
detector = ml.IsolationForestDetector(
    config=IsolationForestConfig(
        n_estimators=100,
        contamination=0.01,
        max_samples=256,
    )
)

# Ensemble detector (combines multiple methods)
detector = ml.EnsembleAnomalyDetector(
    detectors=[
        ml.ZScoreAnomalyDetector(),
        ml.IQRAnomalyDetector(),
    ],
    config=EnsembleConfig(
        strategy=EnsembleStrategy.AVERAGE,  # AVERAGE, MAX, MIN, VOTE, UNANIMOUS
    ),
)

# Fit and predict (use LazyFrame)
df = pl.read_csv("data.csv")
detector.fit(df.lazy())  # pass LazyFrame
result = detector.predict(df.lazy())

print(f"Anomalies: {result.anomaly_count}")
for score in result.get_anomalies():  # use get_anomalies() method
    print(f"  Row {score.index}: {score.anomaly_type.value}")
```

### ML Drift Detection

```python
from truthound import ml
from truthound.ml.drift_detection.distribution import DistributionDriftConfig
from truthound.ml.drift_detection.feature import FeatureDriftConfig

# Distribution drift detector (using Config object)
detector = ml.DistributionDriftDetector(
    config=DistributionDriftConfig(
        method="psi",  # "ks", "psi", "jensen_shannon", "wasserstein"
        threshold=0.05,
        n_bins=10,
    )
)

# Feature drift detector (multi-column)
detector = ml.FeatureDriftDetector(
    config=FeatureDriftConfig(
        threshold=0.05,
        relative_threshold=True,
        alert_on_new_values=True,
    )
)

# Fit baseline (use LazyFrame)
detector.fit(baseline_df.lazy())

# Detect drift with predict()
result = detector.predict(current_df.lazy())
if result.is_drifted:
    drifted_cols = result.get_drifted_columns(threshold=0.05)
    print(f"Drift detected: {drifted_cols}")
```

### Rule Learning

```python
from truthound import ml
from truthound.ml.rule_learning.profile_learner import ProfileLearnerConfig
from truthound.ml.rule_learning.pattern_learner import PatternLearnerConfig
from truthound.ml.rule_learning.constraint_miner import ConstraintMinerConfig

# Learn rules from data (using Config object)
learner = ml.DataProfileRuleLearner(
    config=ProfileLearnerConfig(
        strictness="medium",  # "loose", "medium", "strict"
        min_support=0.1,
        min_confidence=0.8,
        include_range_rules=True,
        include_null_rules=True,
    )
)
learner.fit(df.lazy())
result = learner.predict(df.lazy())

for rule in result.rules:
    print(f"{rule.column}: {rule.rule_type} - {rule.condition}")

# Pattern-based rule learning
learner = ml.PatternRuleLearner(
    config=PatternLearnerConfig(
        min_pattern_ratio=0.9,
        learn_custom_patterns=True,
    )
)
learner.fit(df.lazy())
result = learner.predict(df.lazy())

# Constraint mining
miner = ml.ConstraintMiner(
    config=ConstraintMinerConfig(
        discover_functional_deps=True,
        discover_value_constraints=True,
    )
)
miner.fit(df.lazy())
result = miner.predict(df.lazy())
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
| **Configurations** | `AnomalyConfig`, `StatisticalConfig`, `IsolationForestConfig`, `EnsembleConfig`, `DriftConfig`, `DistributionDriftConfig`, `FeatureDriftConfig`, `ConceptDriftConfig`, `MultivariateDriftConfig`, `RuleLearningConfig`, `ProfileLearnerConfig`, `PatternLearnerConfig`, `ConstraintMinerConfig` |
| **Enums** | `ModelType`, `ModelState`, `AnomalyType`, `SeverityLevel` |
| **Registry** | `ModelRegistry`, `model_registry`, `register_model` |
| **Exceptions** | `MLError`, `ModelNotTrainedError`, `ModelTrainingError`, `ModelLoadError`, `InsufficientDataError` |

---

## Lineage Module

Track data flow and analyze impact of changes.

### Basic Lineage Tracking

```python
from truthound import lineage

# Create tracker (with optional config)
tracker = lineage.LineageTracker()

# Track data sources
tracker.track_source(
    name="raw_customers",
    source_type="file",
    location="/data/customers.csv",
    schema={"id": "Int64", "name": "Utf8", "email": "Utf8"},
    description="Raw customer data",
    owner="data_team",
    tags=["raw", "pii"],
)

tracker.track_source(
    name="raw_orders",
    source_type="table",
    location="postgresql://localhost/db/orders",
)

# Track transformations
tracker.track_transformation(
    name="cleaned_customers",
    sources=["raw_customers"],
    operation="clean",
    location="memory://cleaned_customers",
    description="Removed nulls and normalized emails",
)

tracker.track_transformation(
    name="customer_orders",
    sources=["cleaned_customers", "raw_orders"],
    operation="join",
)

# Track validation
tracker.track_validation(
    name="validated_data",
    sources=["customer_orders"],
    validators=["null", "range", "format"],
)
```

### Impact Analysis

```python
from truthound import lineage

# Use .graph property (not get_graph() method)
analyzer = lineage.ImpactAnalyzer(tracker.graph)

# Forward impact: what depends on this node?
impact = analyzer.analyze_impact("raw_customers")
print(f"Affected nodes: {[n.node.id for n in impact.affected_nodes]}")
print(f"Total affected: {impact.total_affected}")
print(f"Max depth: {impact.max_depth}")

# Get nodes by impact level
critical_nodes = impact.get_by_level(lineage.ImpactLevel.CRITICAL)
high_nodes = impact.get_by_level(lineage.ImpactLevel.HIGH)

# Get summary
print(impact.summary())

# Backward lineage: use get_upstream() on graph
upstream = tracker.graph.get_upstream("validated_data")
print(f"Source nodes: {[n.id for n in upstream]}")

# Find path between nodes
path = tracker.get_path("raw_customers", "validated_data")
if path:
    print(f"Path: {[n.id for n in path]}")
```

### Visualization

```python
from truthound.lineage.visualization import (
    D3Renderer,
    CytoscapeRenderer,
    GraphvizRenderer,
    MermaidRenderer,
    RenderConfig,
    get_renderer,  # factory function
)

# Use .graph property
graph = tracker.graph

# D3.js visualization (interactive HTML)
renderer = D3Renderer(theme="light")  # or "dark"
html = renderer.render_html(graph, RenderConfig(
    layout="hierarchical",  # force, hierarchical, circular, grid
    width=1200,
    height=800,
    orientation="TB",  # TB, BT, LR, RL
    include_metadata=True,
))
with open("lineage.html", "w") as f:
    f.write(html)

# Mermaid diagram (for documentation)
renderer = MermaidRenderer()
mermaid_code = renderer.render(graph, RenderConfig())
print(mermaid_code)

# Graphviz (for static images)
renderer = GraphvizRenderer()
dot = renderer.render(graph, RenderConfig())

# Cytoscape.js (interactive graph)
renderer = CytoscapeRenderer(theme="dark")
html = renderer.render_html(graph, RenderConfig(layout="cose"))

# Factory function for creating renderers
renderer = get_renderer("d3", theme="light")
renderer = get_renderer("mermaid")
```

### OpenLineage Integration

```python
from truthound.lineage.integrations.openlineage import (
    OpenLineageEmitter,
    OpenLineageConfig,
)

# Create emitter with config
emitter = OpenLineageEmitter(
    config=OpenLineageConfig(
        endpoint="http://localhost:5000/api/v1/lineage",
        namespace="my-pipeline",
        producer="truthound",
    )
)

# Start a run
run = emitter.start_run(
    job_name="data-validation",
    inputs=[
        emitter.build_input_dataset(
            name="raw_data",
            namespace="file",
            schema=[{"name": "id", "type": "int"}, {"name": "value", "type": "string"}],
        )
    ],
)

# Emit running status (optional)
emitter.emit_running(run)

# Emit completion with outputs
emitter.emit_complete(
    run,
    outputs=[
        emitter.build_output_dataset(
            name="/data/validated.parquet",
            namespace="file",
            row_count=10000,
        )
    ],
)

# Or emit from existing lineage graph
events = emitter.emit_from_graph(tracker.graph, job_name="etl-pipeline")
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
| **Integration** | `OpenLineageEmitter`, `OpenLineageConfig`, `RunEvent`, `EventType`, `DatasetFacets` |

---

## Real-time Module

Streaming and incremental validation.

### Streaming Validator

```python
from truthound import realtime
from truthound.realtime import StreamingConfig, StreamingMode

# Create streaming validator with config
validator = realtime.StreamingValidator(
    validators=["null", "range", "format"],
    config=StreamingConfig(
        mode=StreamingMode.MICRO_BATCH,
        batch_size=1000,
        batch_timeout_ms=1000,
        error_handling="skip",  # "skip", "fail", "retry"
    ),
)

# Process batches (pass DataFrame, not LazyFrame)
for batch in data_batches:
    result = validator.validate_batch(batch, batch_id="batch_001")
    if result.has_issues:
        for issue in result.issues:
            print(f"Issue: {issue.column} - {issue.issue_type}")
    print(f"Processed: {result.record_count}, Issues: {result.issue_count}")
```

### Incremental Validator

```python
from truthound import realtime
from truthound.realtime import StreamingConfig, WindowConfig, WindowType

# Create incremental validator with state
validator = realtime.IncrementalValidator(
    validators=["unique", "aggregate"],
    config=StreamingConfig(checkpoint_interval_ms=5000),
    window_config=WindowConfig(
        window_type=WindowType.TUMBLING,
        window_size=60,  # seconds
    ),
    state_store=realtime.MemoryStateStore(),
)

# Process increments (use validate_batch method)
for chunk in data_chunks:
    result = validator.validate_batch(chunk)
    print(f"Total rows processed: {validator.total_records}")
    print(f"Total issues: {validator.total_issues}")

# Get aggregate statistics
stats = validator.get_aggregate_stats()
print(f"Batch count: {stats['batch_count']}")
print(f"Issue rate: {stats['issue_rate']:.2%}")
```

### Kafka Integration

```python
from truthound.realtime.adapters.kafka import KafkaAdapter, KafkaAdapterConfig
from truthound.realtime.factory import StreamAdapterFactory
from truthound.realtime import OffsetReset, DeserializationFormat

# Create via factory (pass dict config)
adapter = StreamAdapterFactory.create("kafka", {
    "bootstrap_servers": "localhost:9092",
    "topic": "events",
    "consumer_group": "truthound-validators",
})

# Or direct instantiation with typed config
adapter = KafkaAdapter(
    config=KafkaAdapterConfig(
        bootstrap_servers="localhost:9092",
        topic="events",
        consumer_group="truthound-validators",
        auto_offset_reset=OffsetReset.EARLIEST,
        value_deserializer=DeserializationFormat.JSON,
    )
)

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
from truthound.realtime.adapters.kinesis import KinesisAdapter, KinesisAdapterConfig
from truthound.realtime import DeserializationFormat

adapter = KinesisAdapter(
    config=KinesisAdapterConfig(
        stream_name="my-stream",
        region_name="us-east-1",
        shard_iterator_type="LATEST",  # or "TRIM_HORIZON", "AT_SEQUENCE_NUMBER"
        value_deserializer=DeserializationFormat.JSON,
        # AWS credentials (optional, uses default credential chain)
        # aws_access_key_id="...",
        # aws_secret_access_key="...",
    )
)

async with adapter:
    async for message in adapter.consume():
        process_message(message)
```

### Window-Based Validation

```python
from truthound import realtime
from truthound.realtime import WindowConfig, WindowType, WindowResult
from datetime import datetime, timedelta

# Configure window
window_config = WindowConfig(
    window_type=WindowType.TUMBLING,  # TUMBLING, SLIDING, SESSION, GLOBAL
    window_size=60,  # seconds
    # For SLIDING windows:
    # slide_interval=10,  # slide every 10 seconds
    # For SESSION windows:
    # allowed_lateness=30,  # allow 30s late data
)

# IncrementalValidator supports windowing
validator = realtime.IncrementalValidator(
    validators=["null", "range"],
    window_config=window_config,
    state_store=realtime.MemoryStateStore(),
)

# Process batches and get window results
for batch in data_batches:
    result = validator.validate_batch(batch)

# Get current window result
window_result = validator.get_current_window()
if window_result:
    print(f"Window {window_result.window_id}:")
    print(f"  Total records: {window_result.total_records}")
    print(f"  Total issues: {window_result.total_issues}")
    print(f"  Batch count: {window_result.batch_count}")
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
| **Configs** | `StreamingConfig`, `WindowConfig`, `KafkaAdapterConfig`, `KinesisAdapterConfig` |
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
