# truthound ml anomaly

Detect anomalies in data using statistical and machine learning methods.

## Synopsis

```bash
truthound ml anomaly <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--method` | `-m` | `zscore` | Detection method (zscore, iqr, mad, isolation_forest) |
| `--contamination` | `-c` | `0.1` | Expected anomaly ratio (0.0-0.5) |
| `--columns` | | All numeric | Columns to analyze (comma-separated) |
| `--sample` | `-s` | All rows | Sample size for large datasets |
| `--output` | `-o` | None | Output file path |
| `--format` | `-f` | `console` | Output format (console, json) |

## Description

The `ml anomaly` command detects outliers and unusual data points:

1. **Analyzes** numeric columns for anomalies
2. **Applies** selected detection method
3. **Reports** anomalous rows with scores
4. **Provides** column-level statistics

## Detection Methods

### Z-Score (`zscore`)

Detects values that deviate significantly from the mean.

- **Formula**: `z = (x - μ) / σ`
- **Threshold**: Typically |z| > 3
- **Best for**: Normally distributed data
- **Limitation**: Sensitive to outliers in mean/std calculation

```bash
truthound ml anomaly data.csv --method zscore
```

### Interquartile Range (`iqr`)

Detects values outside the IQR fence.

- **Formula**: `outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR`
- **IQR**: Q3 - Q1 (interquartile range)
- **Best for**: Skewed distributions
- **Advantage**: Robust to extreme outliers

```bash
truthound ml anomaly data.csv --method iqr
```

### Median Absolute Deviation (`mad`)

Detects values using robust median-based statistics.

- **Formula**: `MAD = median(|x - median(x)|)`
- **Threshold**: |x - median| / MAD > 3.5
- **Best for**: Data with extreme outliers
- **Advantage**: Most robust to outliers

```bash
truthound ml anomaly data.csv --method mad
```

### Isolation Forest (`isolation_forest`)

Machine learning-based anomaly detection.

- **Algorithm**: Random forest that isolates anomalies
- **Advantage**: Detects complex, multivariate anomalies
- **Best for**: High-dimensional data, complex patterns
- **Parameter**: `contamination` controls sensitivity

```bash
truthound ml anomaly data.csv --method isolation_forest --contamination 0.05
```

## Examples

### Basic Anomaly Detection

```bash
truthound ml anomaly data.csv
```

Output:
```
Anomaly Detection Results (zscore)
==================================================
Total points: 10000
Anomalies found: 847
Anomaly ratio: 8.47%
Threshold used: 3.0000

Top anomalies:
  Index 4521: score=0.9800, confidence=98.00%
  Index 1234: score=0.9500, confidence=95.00%
  Index 7890: score=0.9200, confidence=92.00%
  ...
```

### Specific Columns

Analyze only selected columns:

```bash
truthound ml anomaly data.csv --columns age,salary,score
```

### Custom Contamination

Adjust expected anomaly ratio:

```bash
# Expect 5% anomalies (more selective)
truthound ml anomaly data.csv --contamination 0.05

# Expect 20% anomalies (more inclusive)
truthound ml anomaly data.csv --contamination 0.2
```

### Statistical Methods

```bash
# Z-score for normal distributions
truthound ml anomaly data.csv --method zscore

# IQR for skewed distributions
truthound ml anomaly data.csv --method iqr

# MAD for robust detection
truthound ml anomaly data.csv --method mad
```

### JSON Output

```bash
truthound ml anomaly data.csv --format json -o anomalies.json
```

Output file (`anomalies.json`):
```json
{
  "file": "data.csv",
  "row_count": 10000,
  "method": "isolation_forest",
  "contamination": 0.1,
  "summary": {
    "total_anomalies": 847,
    "anomaly_ratio": 0.0847
  },
  "column_statistics": [
    {
      "column": "age",
      "anomaly_count": 234,
      "anomaly_ratio": 0.0234,
      "max_score": 0.95
    },
    {
      "column": "salary",
      "anomaly_count": 456,
      "anomaly_ratio": 0.0456,
      "max_score": 0.98
    }
  ],
  "anomalies": [
    {
      "row_index": 4521,
      "score": 0.98,
      "anomalous_columns": ["salary"],
      "values": {"salary": 999999}
    },
    {
      "row_index": 1234,
      "score": 0.95,
      "anomalous_columns": ["age"],
      "values": {"age": -5}
    }
  ]
}
```

## Method Comparison

| Method | Speed | Robustness | Multivariate | Best For |
|--------|-------|------------|--------------|----------|
| zscore | Fast | Low | No | Normal distributions |
| iqr | Fast | Medium | No | Skewed data |
| mad | Fast | High | No | Data with outliers |
| isolation_forest | Medium | High | Yes | Complex patterns |

## Use Cases

### 1. Data Quality Check

```bash
# Find data entry errors
truthound ml anomaly customer_data.csv --method mad --format json -o errors.json
```

### 2. Fraud Detection

```bash
# Detect suspicious transactions
truthound ml anomaly transactions.csv --method isolation_forest --contamination 0.01
```

### 3. Sensor Data Monitoring

```bash
# Find sensor malfunctions
truthound ml anomaly sensor_readings.csv --columns temperature,pressure --method zscore
```

### 4. CI/CD Pipeline

```yaml
# GitHub Actions
- name: Check for Data Anomalies
  run: |
    truthound ml anomaly data.csv --method isolation_forest --format json -o anomalies.json
    # Check if anomaly ratio is too high
    python -c "
    import json
    with open('anomalies.json') as f:
        data = json.load(f)
    if data['summary']['anomaly_ratio'] > 0.15:
        print('Too many anomalies detected!')
        exit(1)
    "
```

## Performance

For large datasets, use the `--sample` option to process a random subset:

```bash
# Sample 100,000 rows for faster processing
truthound ml anomaly large_data.parquet --method isolation_forest --sample 100000

# Isolation Forest may be slow for very large files
# Consider using statistical methods for speed
truthound ml anomaly large_data.parquet --method iqr

# Combine sampling with any method
truthound ml anomaly large_data.parquet --method mad --sample 50000
```

Sampling uses a fixed seed (42) for reproducibility.

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (invalid arguments, file not found, or other error) |

## Related Commands

- [`ml drift`](drift.md) - Detect data drift
- [`ml learn-rules`](learn-rules.md) - Learn validation rules
- [`check`](../core/check.md) - Rule-based validation

## See Also

- [Statistical Methods](../../concepts/statistical-methods.md)
- [Advanced Features](../../concepts/advanced.md)
