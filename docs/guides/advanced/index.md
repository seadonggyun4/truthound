# Advanced Features

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
- **4 Drift Detectors**: Distribution (PSI/KS), Feature, Concept, Multivariate
- **Model Monitoring**: Performance, drift, quality metrics with alerting

```python
from truthound import ml

# Anomaly detection
detector = ml.IsolationForestDetector(contamination=0.1)
detector.fit(train_data)
result = detector.predict(test_data)

# Drift detection
drift = ml.DistributionDriftDetector(method="psi")
drift.fit(reference_data)
result = drift.detect(reference_data, current_data)
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
| Distribution Drift (PSI, KS) | Yes | Yes |
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
