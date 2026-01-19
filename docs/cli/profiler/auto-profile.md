# truthound auto-profile

Generate an advanced statistical profile of data with pattern detection and correlation analysis.

## Synopsis

```bash
truthound auto-profile <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | None | Output file path |
| `--format` | `-f` | `console` | Output format (console, json, yaml) |
| `--patterns/--no-patterns` | | `true` | Enable/disable pattern detection |
| `--correlations/--no-correlations` | | `false` | Enable/disable correlation analysis |
| `--sample` | `-s` | None | Sample size for large datasets |
| `--top-n` | | `10` | Number of top items to display |

## Description

The `auto-profile` command provides advanced data profiling beyond basic statistics:

1. **Statistical Analysis**: Mean, std, percentiles, skewness, kurtosis
2. **Pattern Detection**: Email, phone, URL, date formats, custom patterns
3. **Correlation Analysis**: Numeric column correlations
4. **Distribution Analysis**: Value frequency, entropy, cardinality
5. **Anomaly Detection**: Outlier identification

## Examples

### Basic Profiling

```bash
truthound auto-profile data.csv
```

Output:
```
Advanced Data Profile
=====================
File: data.csv
Rows: 10,000
Columns: 8

Column Analysis
───────────────────────────────────────────────────────────────────
Column        Type      Nulls    Unique    Patterns    Stats
───────────────────────────────────────────────────────────────────
id            Int64     0.0%     100.0%    -           μ=5000.5
email         String    1.0%     99.8%     email       -
phone         String    2.5%     95.0%     phone       -
age           Int64     5.0%     0.7%      -           μ=35.2 σ=12.8
salary        Float64   3.0%     45.0%     -           μ=65432 σ=15234
category      String    0.0%     0.05%     -           5 values
───────────────────────────────────────────────────────────────────

Top Patterns Detected:
  email: email (confidence: 98%)
  phone: phone_us (confidence: 95%)
```

### With Correlation Analysis

```bash
truthound auto-profile data.csv --correlations
```

Additional output:
```
Correlation Matrix (numeric columns)
────────────────────────────────────────
           age      salary    tenure
age        1.00     0.45      0.72
salary     0.45     1.00      0.38
tenure     0.72     0.38      1.00
────────────────────────────────────────
```

### Disable Pattern Detection

For faster profiling without pattern analysis:

```bash
truthound auto-profile data.csv --no-patterns
```

### Sample Large Datasets

Profile a sample of large datasets:

```bash
truthound auto-profile large_data.parquet --sample 100000
```

### JSON Output

```bash
truthound auto-profile data.csv --format json -o profile.json
```

Output file:
```json
{
  "file": "data.csv",
  "row_count": 10000,
  "column_count": 8,
  "columns": [
    {
      "name": "email",
      "dtype": "String",
      "null_ratio": 0.01,
      "unique_ratio": 0.998,
      "patterns": [
        {
          "type": "email",
          "confidence": 0.98,
          "match_ratio": 0.97
        }
      ],
      "top_values": [
        {"value": "john@example.com", "count": 5},
        {"value": "jane@test.org", "count": 3}
      ]
    },
    {
      "name": "age",
      "dtype": "Int64",
      "null_ratio": 0.05,
      "statistics": {
        "mean": 35.2,
        "std": 12.8,
        "min": 18,
        "max": 85,
        "q25": 26,
        "median": 34,
        "q75": 44,
        "skewness": 0.45,
        "kurtosis": -0.32
      }
    }
  ],
  "correlations": {
    "age_salary": 0.45,
    "age_tenure": 0.72
  }
}
```

### YAML Output

```bash
truthound auto-profile data.csv --format yaml -o profile.yaml
```

### Custom Top-N

Show top 20 values per column:

```bash
truthound auto-profile data.csv --top-n 20
```

## Profile Contents

### Basic Metrics

| Metric | Description |
|--------|-------------|
| `row_count` | Total number of rows |
| `column_count` | Total number of columns |
| `memory_usage` | Estimated memory usage |

### Column Metrics

| Metric | Description | Types |
|--------|-------------|-------|
| `dtype` | Data type | All |
| `null_count` | Number of null values | All |
| `null_ratio` | Proportion of nulls | All |
| `unique_count` | Number of unique values | All |
| `unique_ratio` | Proportion of unique values | All |
| `top_values` | Most frequent values | All |

### Numeric Statistics

| Metric | Description |
|--------|-------------|
| `mean` | Arithmetic mean |
| `std` | Standard deviation |
| `min` / `max` | Range |
| `q25` / `median` / `q75` | Quartiles |
| `skewness` | Distribution asymmetry |
| `kurtosis` | Distribution tailedness |
| `entropy` | Information entropy |

### Pattern Detection

Automatically detected patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| `email` | Email addresses | `john@example.com` |
| `phone` | Phone numbers | `+1-555-123-4567` |
| `phone_us` | US phone format | `(555) 123-4567` |
| `url` | URLs | `https://example.com` |
| `ip_address` | IP addresses | `192.168.1.1` |
| `date_iso` | ISO date format | `2024-01-15` |
| `uuid` | UUID format | `550e8400-e29b-41d4-a716-446655440000` |
| `credit_card` | Credit card numbers | `4111-1111-1111-1111` |
| `ssn` | US SSN | `123-45-6789` |

## Use Cases

### 1. Data Discovery

Understand unknown datasets:

```bash
truthound auto-profile new_dataset.csv --patterns --correlations --format json -o discovery.json
```

### 2. Pre-Processing Analysis

Identify data quality issues before processing:

```bash
truthound auto-profile raw_data.csv
# Check for high null ratios, unexpected patterns, etc.
```

### 3. Feature Engineering

Identify correlated features:

```bash
truthound auto-profile features.csv --correlations
```

### 4. Profile for Rule Generation

Generate profile for validation suite creation:

```bash
# Step 1: Profile
truthound auto-profile data.csv --format json -o profile.json

# Step 2: Generate rules
truthound generate-suite profile.json -o rules.yaml
```

## Comparison with `profile`

| Feature | `profile` | `auto-profile` |
|---------|-----------|----------------|
| Basic statistics | Yes | Yes |
| Pattern detection | Basic | Advanced |
| Correlation analysis | No | Yes |
| Sampling | No | Yes |
| Top-N values | No | Yes |
| Output formats | console, json | console, json, yaml |
| Performance | Faster | More detailed |

## Related Commands

- [`profile`](../core/profile.md) - Basic data profiling
- [`generate-suite`](generate-suite.md) - Generate rules from profile
- [`quick-suite`](quick-suite.md) - Profile + generate in one step

## See Also

- [Python API: th.profile()](../../python-api/core-functions.md#thprofile)
- [Data Profiling Guide](../../guides/profiler.md)
