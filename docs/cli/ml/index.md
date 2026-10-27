# ML Commands

Machine learning-based data quality detection commands.

## Native model configuration and failures

Python `truthound.ml` model constructors capture an independent copy of supplied
configuration and mutable overrides. Keyword overrides do not mutate the caller's
config, and unknown setting names raise `ValueError` instead of being ignored.
The model's own `config` remains its explicit mutable configuration surface.
Concept-drift training rejects missing selected features and propagates failed
correlation computations as `ModelTrainingError`. Detection rejects missing target
or selected trained features; it does not turn these errors into zero correlation
or a successful no-drift verdict. This contract also applies through selected MCP
model constructors and detection operations.


Model restore stages a separate copy of native learned state, including ensemble
children, before publishing it to the live owner. Invalid files preserve a trained
model and its configuration. Custom model state must support copying for staged
restore. Saving uses atomic sibling publication; pre-replacement failures preserve
the previous artifact. A directory synchronization failure after replacement
requires inspecting the published file. Use one writer per model owner.


## Overview

| Command | Description | Primary Use Case |
|---------|-------------|------------------|
| [`anomaly`](anomaly.md) | Detect anomalies in data | Outlier detection |
| [`drift`](drift.md) | Detect data drift | Model monitoring |
| [`learn-rules`](learn-rules.md) | Learn validation rules from data | Rule automation |

## What is ML-based Detection?

ML-based detection uses statistical and machine learning algorithms to:

- **Detect anomalies**: Find outliers and unusual patterns
- **Detect drift**: Identify distribution changes between datasets
- **Learn rules**: Automatically generate validation rules from data

## Workflow

```mermaid
graph LR
    A[Data] --> B{Task?}
    B -->|Outliers| C[ml anomaly]
    B -->|Changes| D[ml drift]
    B -->|Rules| E[ml learn-rules]
    C --> F[Anomaly Report]
    D --> G[Drift Report]
    E --> H[Validation Rules]
```

## Quick Examples

### Anomaly Detection

```bash
# Detect outliers using Isolation Forest
truthound ml anomaly data.csv --method isolation_forest

# Detect outliers in specific columns
truthound ml anomaly data.csv --columns age,salary --method zscore
```

### Drift Detection

```bash
# Detect distribution drift
truthound ml drift baseline.csv current.csv --method distribution

# Detect multivariate drift
truthound ml drift train.csv production.csv --method multivariate
```

### Rule Learning

```bash
# Learn validation rules from data
truthound ml learn-rules data.csv -o rules.json --strictness medium
```

## Detection Methods

### Anomaly Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `zscore` | Z-score based detection | Normal distributions |
| `iqr` | Interquartile range | Robust to outliers |
| `mad` | Median absolute deviation | Skewed distributions |
| `isolation_forest` | ML-based isolation | Complex patterns |

### Drift Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `distribution` | Per-column distribution comparison | Feature drift |
| `feature` | Feature-wise statistical tests | ML features |
| `multivariate` | Multi-dimensional drift detection | Correlated features |

## Use Cases

### 1. Data Quality Monitoring

```bash
# Scheduled anomaly check
truthound ml anomaly daily_data.csv --method isolation_forest --format json -o anomalies.json
```

### 2. ML Model Monitoring

```bash
# Check for feature drift before retraining
truthound ml drift training_data.csv production_data.csv --method multivariate --threshold 0.05
```

### 3. Automated Rule Generation

```bash
# Bootstrap validation rules from reference data
truthound ml learn-rules reference_data.csv -o rules.json --strictness strict
```

### 4. CI/CD Integration

```yaml
# GitHub Actions
- name: Check for Data Drift
  run: |
    truthound ml drift baseline.csv current.csv --threshold 0.1
    if [ $? -ne 0 ]; then
      echo "Data drift detected!"
      exit 1
    fi
```

## Performance Considerations

| Method | Speed | Memory | Scalability |
|--------|-------|--------|-------------|
| zscore | Fast | Low | Excellent |
| iqr | Fast | Low | Excellent |
| mad | Fast | Low | Excellent |
| isolation_forest | Medium | Medium | Good |
| distribution drift | Fast | Low | Excellent |
| multivariate drift | Slow | High | Limited |

For large datasets, consider:

```bash
# Use sampling for large files
truthound ml anomaly large_data.parquet --method isolation_forest --sample 100000
```

## Next Steps

- [anomaly](anomaly.md) - Detect anomalies
- [drift](drift.md) - Detect data drift
- [learn-rules](learn-rules.md) - Learn validation rules

## See Also

- [Statistical Methods](../../concepts/statistical-methods.md)
- [Advanced Features](../../concepts/advanced.md)
- [compare command](../core/compare.md) - Statistical drift detection

Native model `sample_size` selects up to that many rows using seeded row-hash priorities from Core sampling, rather than an input prefix. `random_seed` must be an unsigned 64-bit integer; `sample_size` must be positive or `None`. Repeatability requires identical input order and the same Polars version; this is not a hard memory limit because the native plan orders row priorities. Concept streaming methods use their separately documented tail windows.

Built-in anomaly and drift training stages independent candidate state and publishes it only after complete success. A failed refit preserves a previously trained model. Ensembles own copies of supplied children; successful training does not train the caller’s original objects. Weights must match children, be finite and nonnegative, and have a finite positive total. Strategy and vote threshold are validated. Adding or removing a child invalidates training and requires `fit()` again. Failed ensemble training preserves every previous child.
