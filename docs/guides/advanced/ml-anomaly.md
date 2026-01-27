# ML Anomaly Detection

Truthound provides comprehensive ML-based anomaly detection, drift detection, and model monitoring capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Anomaly Detection](#anomaly-detection)
3. [Drift Detection](#drift-detection)
4. [Model Monitoring](#model-monitoring)
5. [Rule Learning](#rule-learning)
6. [Configuration Reference](#configuration-reference)

---

## Overview

The ML module (`truthound.ml`) provides:

- **6 Anomaly Detectors**: Statistical and tree-based algorithms
- **4 Drift Detectors**: Distribution, feature, concept, and multivariate drift
- **Model Monitoring**: Performance, drift, and quality metrics with alerting
- **Rule Learning**: Automatic validation rule generation from data profiles

**Location**: `src/truthound/ml/`

```
ml/
├── __init__.py              # Public API exports
├── base.py                  # Core abstractions (1,183 lines)
├── anomaly_models/          # Anomaly detection algorithms
├── drift_detection/         # Drift detection methods
├── monitoring/              # ML model monitoring
└── rule_learning/           # Automatic rule generation
```

---

## Anomaly Detection

### Supported Algorithms

| Detector | Category | Algorithm | Use Case |
|----------|----------|-----------|----------|
| `ZScoreAnomalyDetector` | Statistical | Z-score (std devs from mean) | Normally distributed data |
| `IQRAnomalyDetector` | Statistical | Interquartile range | Robust to mild outliers |
| `MADAnomalyDetector` | Statistical | Median Absolute Deviation | Extremely robust to outliers |
| `IsolationForestDetector` | Tree-based | Isolation Forest | Non-linear, high dimensions |
| `EnsembleAnomalyDetector` | Meta | Multiple strategy voting | Best coverage |

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

**Z-Score Detector**

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

**Z-Score Computation:**
```
z_score = |value - mean| / std
normalized_score = min(1.0, z_score / threshold)
```

**IQR Computation:**
```
bounds = [Q1 - k*IQR, Q3 + k*IQR]  where k=1.5 (default)
normalized_score = min(1.0, distance_from_bounds / IQR)
```

**MAD Computation:**
```
MAD = median(|Xi - median(X)|)
modified_z = 0.6745 * |value - median| / MAD
normalized_score = min(1.0, modified_z / threshold)
```

### Isolation Forest Detector

Pure Python implementation of Isolation Forest algorithm.

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

**Score Computation:**
```
score = 2^(-E(h(x)) / c(n))

Where:
- E(h(x)) = average path length to isolate sample
- c(n) = average path length for unsuccessful search
- c(n) = 2 * (ln(n-1) + 0.5772) - (2 * (n-1) / n)
```

### Ensemble Detector

Combine multiple detectors with different strategies.

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

**Ensemble Strategies:**

| Strategy | Description |
|----------|-------------|
| `AVERAGE` | Simple arithmetic mean of scores |
| `WEIGHTED_AVERAGE` | Weighted by detector reliability |
| `MAX` | Conservative - maximum score across detectors |
| `MIN` | Very conservative - minimum score |
| `VOTE` | Majority voting with threshold |
| `UNANIMOUS` | All detectors must agree |

### Result Types

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

**Anomaly Types:**

| Type | Description |
|------|-------------|
| `POINT` | Single point anomaly |
| `CONTEXTUAL` | Anomaly in context (normal value, wrong time) |
| `COLLECTIVE` | Group of related anomalies |
| `PATTERN` | Pattern violation |
| `TREND` | Trend deviation |
| `SEASONAL` | Seasonal violation |

---

## Drift Detection

### Quick Drift Detection with th.compare()

For simple drift detection between two datasets, use `th.compare()` with 14 available methods:

```python
import truthound as th

# Auto-select best method based on column type
drift = th.compare("baseline.csv", "current.csv", method="auto")

# Statistical tests (numeric columns)
drift = th.compare("baseline.csv", "current.csv", method="ks")          # Kolmogorov-Smirnov
drift = th.compare("baseline.csv", "current.csv", method="psi")         # Population Stability Index
drift = th.compare("baseline.csv", "current.csv", method="cvm")         # Cramér-von Mises
drift = th.compare("baseline.csv", "current.csv", method="anderson")    # Anderson-Darling

# Divergence metrics
drift = th.compare("baseline.csv", "current.csv", method="kl")          # Kullback-Leibler divergence
drift = th.compare("baseline.csv", "current.csv", method="js")          # Jensen-Shannon divergence

# Distance metrics (any column type)
drift = th.compare("baseline.csv", "current.csv", method="wasserstein") # Earth Mover's Distance
drift = th.compare("baseline.csv", "current.csv", method="hellinger")   # Hellinger distance (bounded)
drift = th.compare("baseline.csv", "current.csv", method="bhattacharyya") # Bhattacharyya distance
drift = th.compare("baseline.csv", "current.csv", method="tv")          # Total Variation distance

# Distance metrics (numeric only)
drift = th.compare("baseline.csv", "current.csv", method="energy")      # Energy distance
drift = th.compare("baseline.csv", "current.csv", method="mmd")         # Maximum Mean Discrepancy

# For categorical columns
drift = th.compare("baseline.csv", "current.csv", method="chi2")        # Chi-squared

# Check results
if drift.has_drift:
    for col in drift.columns:
        if col.result.drifted:
            print(f"{col.column}: {col.result.method} = {col.result.statistic:.4f}")
```

See [Python API: th.compare()](../../python-api/core-functions.md#thcompare) for full documentation.

### ML-Based Drift Detection

For advanced ML-based drift detection with model monitoring capabilities:

| Detector | Type | Methods | Use Case |
|----------|------|---------|----------|
| `DistributionDriftDetector` | Univariate | PSI, KS, Jensen-Shannon, Wasserstein | Input distribution changes |
| `FeatureDriftDetector` | Feature-level | Per-feature statistics | Individual feature monitoring |
| `ConceptDriftDetector` | Concept | DDM, ADWIN, Page-Hinkley | Model decision boundary shifts |
| `MultivariateDriftDetector` | Multivariate | PCA, Correlation, Mahalanobis | Joint distribution changes |

### Distribution Drift Detection

Detects changes in univariate distributions.

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

**Distribution Methods:**

| Method | Description | Interpretation |
|--------|-------------|----------------|
| `psi` | Population Stability Index | <0.1 stable, 0.1-0.25 small drift, >0.25 significant |
| `ks` | Kolmogorov-Smirnov test | p-value based significance |
| `jensen_shannon` | Symmetric KL divergence | 0-1 range, 0 = identical |
| `wasserstein` | Earth-Mover distance | Minimum transport cost |
| `kl` | Kullback-Leibler divergence | Information loss measurement |
| `cvm` | Cramér-von Mises test | More sensitive to tails than KS |
| `anderson` | Anderson-Darling test | Most sensitive to tail differences |
| `hellinger` | Hellinger distance | Bounded [0,1], true metric |
| `bhattacharyya` | Bhattacharyya distance | Classification error bounds |
| `tv` | Total Variation distance | Max probability difference |
| `energy` | Energy distance | Location/scale sensitivity |
| `mmd` | Maximum Mean Discrepancy | High-dimensional kernel-based |

> **Note:** For quick drift checks, use `th.compare()` instead. The ML module is better suited for continuous monitoring and model training workflows.

### Feature Drift Detection

Monitors per-feature statistics for drift.

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

**Tracked Statistics:**

- **Numeric**: mean, std, min, max, Q1, Q2, Q3, null_ratio, skewness, kurtosis
- **Categorical**: value distribution, new unique values, category shifts

### Concept Drift Detection

Detects changes in feature-target relationships.

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

**Concept Drift Methods:**

| Method | Description |
|--------|-------------|
| `ddm` | Drift Detection Method - monitors error rate |
| `adwin` | Adaptive Windowing - auto-adjusts window size |
| `page_hinkley` | Cumulative sum - detects small gradual changes |

### Multivariate Drift Detection

Detects joint distribution changes.

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

**Multivariate Methods:**

| Method | Description |
|--------|-------------|
| `correlation` | Frobenius norm of correlation matrix difference |
| `pca` | Changes in explained variance ratios |
| `mahalanobis` | Covariance-aware distance metric |

---

## Model Monitoring

### ModelMonitor

Unified monitoring interface for ML models in production.

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

### Alert Rules

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

### Alert Handlers

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

---

## Rule Learning

### DataProfileRuleLearner

Automatically generate validation rules from data profiles.

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

**Generated Rule Types:**

| Rule Type | Description |
|-----------|-------------|
| Range | Min/max bounds for numeric columns |
| Null | Completeness requirements |
| Uniqueness | Primary key detection |
| Type | Data type validation |
| Format | Regex patterns, length constraints |

---

## Configuration Reference

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

---

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

---

## Model Persistence

```python
# Save trained model
detector.save("my_detector.json")

# Load model
loaded = IsolationForestDetector.load("my_detector.json")

# Use loaded model
result = loaded.predict(new_data)
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Statistical fit | O(n) | Single pass |
| Statistical score | O(m) | m = test rows |
| Isolation Forest fit | O(n log n * k) | k = n_estimators |
| Isolation Forest score | O(m * k * d) | d = tree depth |
| Distribution drift | O(n + m) | Both datasets |
| Feature drift | O(n + m) * cols | Per-column |

---

## Thread Safety

- All models use `threading.RLock()` for concurrent access
- Model state machine tracks lifecycle: `UNTRAINED → TRAINING → TRAINED → READY`
- Global tracker can be set/retrieved with thread-safe locks

---

## See Also

- [Drift Detection](../validators/index.md) - Validator-based drift detection
- [Model Monitoring](../ci-cd/index.md) - CI/CD integration
- [Performance Tuning](performance.md) - Optimization strategies
