# truthound compare

Compare two datasets and detect data drift. This command identifies distributional changes between a baseline and current dataset.

## Synopsis

```bash
truthound compare <baseline> <current> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `baseline` | Yes | Path to the baseline (reference) data file |
| `current` | Yes | Path to the current data file to compare |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--columns` | `-c` | All | Comma-separated columns to compare |
| `--method` | `-m` | `auto` | Detection method (auto, ks, psi, chi2, js) |
| `--threshold` | `-t` | Auto | Custom drift threshold |
| `--format` | `-f` | `console` | Output format (console, json) |
| `--output` | `-o` | None | Output file path |
| `--strict` | | `false` | Exit with code 1 if drift detected |

## Description

The `compare` command detects data drift by comparing statistical distributions:

1. **Distribution Comparison**: Compares value distributions between datasets
2. **Statistical Tests**: Applies appropriate tests based on data types
3. **Threshold Evaluation**: Determines if changes are significant
4. **Column-by-Column Analysis**: Reports drift for each column

Data drift can indicate:
- Data quality issues in the pipeline
- Changes in source systems
- Model performance degradation risk
- Schema or business logic changes

## Examples

### Basic Comparison

```bash
truthound compare baseline.csv current.csv
```

Output:
```
Drift Detection Report
======================
Baseline: baseline.csv (10,000 rows)
Current: current.csv (12,000 rows)

Column Results
──────────────────────────────────────────────────────────────────
Column          Method    Score     Threshold   Drift Detected
──────────────────────────────────────────────────────────────────
age             KS        0.045     0.05        No
income          KS        0.156     0.05        Yes ⚠️
category        Chi2      0.023     0.05        No
region          Chi2      0.234     0.05        Yes ⚠️
score           PSI       0.089     0.10        No
──────────────────────────────────────────────────────────────────

Summary:
  Total Columns: 5
  Drift Detected: 2
  Status: DRIFT DETECTED
```

### Specific Columns

Compare only selected columns:

```bash
truthound compare baseline.csv current.csv -c age,income,score
```

### Specific Method

Force a specific detection method:

```bash
# Kolmogorov-Smirnov test
truthound compare baseline.csv current.csv --method ks

# Population Stability Index
truthound compare baseline.csv current.csv --method psi

# Chi-squared test (for categorical)
truthound compare baseline.csv current.csv --method chi2

# Jensen-Shannon divergence
truthound compare baseline.csv current.csv --method js
```

### Custom Threshold

Set a custom drift threshold:

```bash
truthound compare baseline.csv current.csv --threshold 0.1
```

### Strict Mode (CI/CD)

Exit with code 1 if drift is detected:

```bash
truthound compare baseline.csv current.csv --strict
```

### JSON Output

```bash
truthound compare baseline.csv current.csv --format json -o drift_report.json
```

Output file:
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
  "has_drift": true,
  "column_results": [
    {
      "column": "age",
      "method": "ks",
      "score": 0.045,
      "threshold": 0.05,
      "p_value": 0.23,
      "has_drift": false
    },
    {
      "column": "income",
      "method": "ks",
      "score": 0.156,
      "threshold": 0.05,
      "p_value": 0.001,
      "has_drift": true
    }
  ],
  "summary": {
    "total_columns": 5,
    "drift_columns": 2,
    "drift_ratio": 0.4
  }
}
```

## Detection Methods

### Automatic Selection (`auto`)

The default `auto` method selects the best test based on data type:

| Data Type | Method Selected |
|-----------|-----------------|
| Continuous numeric | Kolmogorov-Smirnov (KS) |
| Categorical (few values) | Chi-squared |
| Categorical (many values) | Jensen-Shannon |
| Binary | Chi-squared |

### Kolmogorov-Smirnov (`ks`)

Best for: Continuous numeric data

- Tests if two samples come from the same distribution
- Returns D-statistic (max distance between CDFs) and p-value
- Default threshold: 0.05 (D-statistic)

```bash
truthound compare data1.csv data2.csv --method ks
```

### Population Stability Index (`psi`)

Best for: Model monitoring, production ML

- Measures population shift between bins
- Industry standard for model monitoring
- Interpretation:
  - PSI < 0.1: No significant change
  - PSI 0.1-0.25: Moderate change
  - PSI > 0.25: Significant change

```bash
truthound compare train.csv production.csv --method psi
```

### Chi-Squared (`chi2`)

Best for: Categorical data

- Tests independence of categorical distributions
- Returns χ² statistic and p-value
- Requires sufficient sample size per category

```bash
truthound compare categories_old.csv categories_new.csv --method chi2
```

### Jensen-Shannon Divergence (`js`)

Best for: Any distribution, robust to small samples

- Symmetric measure of distribution similarity
- Range: 0 (identical) to 1 (completely different)
- Works for both numeric and categorical data

```bash
truthound compare data1.csv data2.csv --method js
```

## Method Comparison

| Method | Numeric | Categorical | Sensitivity | Use Case |
|--------|---------|-------------|-------------|----------|
| KS | Excellent | Poor | High | Statistical testing |
| PSI | Good | Good | Medium | ML monitoring |
| Chi2 | Poor | Excellent | Medium | Category changes |
| JS | Good | Good | Low | General purpose |

## Use Cases

### 1. ML Model Monitoring

Detect feature drift in production:

```bash
# Compare training data to production
truthound compare training_data.csv production_data.csv --method psi --strict
```

### 2. Data Pipeline Validation

Validate daily data against baseline:

```bash
# In CI/CD pipeline
truthound compare baseline/data.csv daily/$(date +%Y%m%d).csv --strict
```

### 3. A/B Test Validation

Ensure test groups are comparable:

```bash
truthound compare control_group.csv treatment_group.csv -c demographic_features
```

### 4. Schema Evolution Detection

Detect changes in data distribution:

```bash
truthound compare last_week.parquet this_week.parquet --format json -o weekly_drift.json
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Check Data Drift
  run: |
    truthound compare baseline.csv production.csv --strict --format json -o drift.json

- name: Upload Drift Report
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: drift-report
    path: drift.json
```

### Scheduled Monitoring

```yaml
# Run daily drift check
schedule:
  - cron: '0 6 * * *'

jobs:
  drift-check:
    steps:
      - name: Download baseline
        run: aws s3 cp s3://bucket/baseline.parquet baseline.parquet

      - name: Download latest data
        run: aws s3 cp s3://bucket/latest.parquet latest.parquet

      - name: Check drift
        run: truthound compare baseline.parquet latest.parquet --strict
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (no drift, or drift found without `--strict`) |
| 1 | Drift detected with `--strict` flag |
| 2 | Usage error (invalid arguments) |

## Related Commands

- [`check`](check.md) - Validate data quality
- [`profile`](profile.md) - Generate data profile
- [`ml drift`](../ml/drift.md) - Advanced ML-based drift detection

## See Also

- [Python API: th.compare()](../../python-api/core-functions.md#thcompare)
- [Statistical Methods](../../concepts/statistical-methods.md)
- [ML Drift Detection](../ml/drift.md)
