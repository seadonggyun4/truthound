# truthound ml drift

Detect data drift between two datasets using machine learning methods.

## Synopsis

```bash
truthound ml drift <baseline> <current> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `baseline` | Yes | Path to the baseline (reference) data file (CSV, JSON, Parquet, NDJSON, JSONL) |
| `current` | Yes | Path to the current data file to compare (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--method` | `-m` | `feature` | Detection method (distribution, feature, multivariate) |
| `--threshold` | `-t` | `0.1` | Drift threshold (0.0-1.0) |
| `--columns` | | All | Columns to compare (comma-separated) |
| `--output` | `-o` | None | Output file path |

## Description

The `ml drift` command detects changes in data distribution using ML methods:

1. **Compares** statistical properties between datasets
2. **Detects** significant distribution shifts
3. **Reports** drift scores per column
4. **Identifies** most affected features

## Detection Methods

### Distribution (`distribution`)

Compares individual column distributions.

- **Tests**: KS-test (numeric), Chi-squared (categorical)
- **Best for**: Feature-level drift detection
- **Speed**: Fast
- **Interpretability**: High

```bash
truthound ml drift baseline.csv current.csv --method distribution
```

### Feature (`feature`)

Statistical feature comparison with multiple tests.

- **Tests**: Mean/std shift, distribution tests, correlation changes
- **Best for**: ML feature monitoring
- **Speed**: Fast
- **Interpretability**: High

```bash
truthound ml drift baseline.csv current.csv --method feature
```

### Multivariate (`multivariate`)

Detects drift in the joint distribution.

- **Algorithm**: Compares multivariate distributions
- **Best for**: Correlated features, complex drift patterns
- **Speed**: Slower
- **Advantage**: Catches drift that single-column tests miss

```bash
truthound ml drift baseline.csv current.csv --method multivariate
```

## Examples

### Basic Drift Detection

```bash
truthound ml drift baseline.csv current.csv
```

Output:
```
ML Drift Detection Report
=========================
Baseline: baseline.csv (10,000 rows)
Current: current.csv (12,000 rows)
Method: distribution
Threshold: 0.1

Overall Drift Score: 0.23 (DRIFT DETECTED)

Column Results
──────────────────────────────────────────────────────────────────
Column          Score     Threshold   Drift Detected    Details
──────────────────────────────────────────────────────────────────
age             0.05      0.10        No                -
income          0.34      0.10        Yes ⚠️            Mean shift: +15%
category        0.08      0.10        No                -
region          0.42      0.10        Yes ⚠️            New values: 3
score           0.12      0.10        Yes ⚠️            Std shift: +22%
──────────────────────────────────────────────────────────────────

Drift Summary:
  Total Columns: 5
  Drifted Columns: 3 (60%)
  Status: DRIFT DETECTED

Recommendations:
  - Review 'income' for mean shift (+15%)
  - Check 'region' for new category values
  - Investigate 'score' distribution change
```

### Specific Columns

Compare only selected columns:

```bash
truthound ml drift baseline.csv current.csv --columns age,income,score
```

### Custom Threshold

Adjust drift sensitivity:

```bash
# More sensitive (lower threshold)
truthound ml drift baseline.csv current.csv --threshold 0.05

# Less sensitive (higher threshold)
truthound ml drift baseline.csv current.csv --threshold 0.2
```

### Multivariate Detection

Detect complex drift patterns:

```bash
truthound ml drift train_data.csv production_data.csv --method multivariate
```

Output:
```
ML Drift Detection Report
=========================
Method: multivariate

Multivariate Drift Analysis
──────────────────────────────────────────────────────────────────
Metric                    Value       Threshold   Status
──────────────────────────────────────────────────────────────────
Overall Drift Score       0.28        0.10        DRIFT DETECTED
Correlation Change        0.15        0.10        Changed
Covariance Shift          0.22        0.10        Shifted
Principal Component Drift 0.18        0.10        Drifted
──────────────────────────────────────────────────────────────────

Most Affected Feature Combinations:
  1. income × tenure: correlation changed from 0.72 to 0.45
  2. age × salary: distribution shift detected
  3. region × category: joint distribution changed
```

### JSON Output

```bash
truthound ml drift baseline.csv current.csv --output drift_report.json
```

Output file (`drift_report.json`):
```json
{
  "baseline": {
    "file": "baseline.csv",
    "rows": 10000
  },
  "current": {
    "file": "current.csv",
    "rows": 12000
  },
  "method": "distribution",
  "threshold": 0.1,
  "overall_drift_score": 0.23,
  "has_drift": true,
  "column_results": [
    {
      "column": "age",
      "score": 0.05,
      "has_drift": false,
      "details": null
    },
    {
      "column": "income",
      "score": 0.34,
      "has_drift": true,
      "details": {
        "type": "mean_shift",
        "baseline_mean": 50000,
        "current_mean": 57500,
        "percent_change": 15.0
      }
    },
    {
      "column": "region",
      "score": 0.42,
      "has_drift": true,
      "details": {
        "type": "new_categories",
        "new_values": ["west", "southwest", "northwest"],
        "count": 3
      }
    }
  ],
  "summary": {
    "total_columns": 5,
    "drifted_columns": 3,
    "drift_ratio": 0.6
  }
}
```

## Method Comparison

| Method | Speed | Complexity | Correlated Features | Best For |
|--------|-------|------------|---------------------|----------|
| distribution | Fast | Low | No | Quick checks |
| feature | Fast | Medium | No | ML monitoring |
| multivariate | Slow | High | Yes | Complex drift |

## Comparison: `ml drift` vs `compare`

| Feature | `ml drift` | `compare` |
|---------|------------|-----------|
| Focus | ML-oriented drift detection | Statistical testing |
| Methods | distribution, feature, multivariate | ks, psi, chi2, js |
| Multivariate | Yes | No |
| Speed | Varies | Fast |
| Best for | ML pipelines | General use |

## Use Cases

### 1. ML Model Monitoring

```bash
# Check if production data has drifted from training
truthound ml drift training_data.csv production_data.csv --method multivariate --threshold 0.1
```

### 2. Feature Store Monitoring

```bash
# Monitor feature drift daily
truthound ml drift features_yesterday.parquet features_today.parquet --method feature
```

### 3. A/B Test Validation

```bash
# Ensure test groups are comparable
truthound ml drift control_group.csv treatment_group.csv --columns demographics
```

### 4. CI/CD Pipeline

```yaml
# GitHub Actions
- name: Check Feature Drift
  run: |
    truthound ml drift baseline.csv current.csv --method multivariate --threshold 0.15 --output drift.json
    # Parse result and fail if drift detected
    python -c "
    import json
    with open('drift.json') as f:
        result = json.load(f)
    if result['has_drift']:
        print(f'Drift detected! Score: {result[\"overall_drift_score\"]}')
        for col in result['column_results']:
            if col['has_drift']:
                print(f'  - {col[\"column\"]}: {col[\"score\"]}')
        exit(1)
    "
```

### 5. Retraining Trigger

```bash
# Check drift and trigger retraining if needed
if truthound ml drift train.csv prod.csv --method multivariate --threshold 0.2; then
  echo "No significant drift"
else
  echo "Drift detected, triggering retraining"
  python retrain_model.py
fi
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (invalid arguments, file not found, or other error) |

> **Note**: Drift detection results are reported in the output, but do not affect the exit code. Use `--output result.json` to save JSON output and parse the `has_drift` field for CI/CD decisions.

## Related Commands

- [`compare`](../core/compare.md) - Statistical drift detection
- [`ml anomaly`](anomaly.md) - Anomaly detection
- [`ml learn-rules`](learn-rules.md) - Learn validation rules

## See Also

- [Statistical Methods](../../concepts/statistical-methods.md)
- [Advanced Features](../../concepts/advanced.md)
- [ML Model Monitoring](../../guides/ml-monitoring.md)
