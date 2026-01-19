# truthound profile

Generate a statistical profile of the data. This command provides summary statistics and insights about your dataset.

## Synopsis

```bash
truthound profile <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `console` | Output format (console, json) |
| `--output` | `-o` | None | Output file path |

## Description

The `profile` command analyzes your data and generates comprehensive statistics:

- **Basic Statistics**: Row count, column count, memory usage
- **Column Statistics**: Data types, null ratios, unique counts
- **Numeric Analysis**: Mean, std, min, max, percentiles
- **Distribution Info**: Skewness, kurtosis, entropy
- **Pattern Detection**: Common data patterns

## Examples

### Basic Profiling

```bash
truthound profile data.csv
```

Output:
```
Data Profile Report
===================
File: data.csv
Rows: 10,000
Columns: 8
Memory: 2.4 MB

Column Statistics
─────────────────────────────────────────────────────────────────────────────
Column        Type      Nulls     Unique    Mean        Std         Min/Max
─────────────────────────────────────────────────────────────────────────────
id            Int64     0.0%      100.0%    5000.5      2886.9      1 / 10000
name          String    2.5%      85.0%     -           -           -
email         String    1.0%      99.8%     -           -           -
age           Int64     5.0%      0.7%      35.2        12.8        18 / 85
salary        Float64   3.0%      45.0%     65432.1     15234.5     30000 / 150000
department    String    0.0%      0.05%     -           -           -
join_date     Date      0.0%      12.5%     -           -           2020-01-01 / 2024-12-31
is_active     Boolean   0.0%      0.02%     -           -           -
─────────────────────────────────────────────────────────────────────────────
```

### JSON Output

```bash
truthound profile data.csv --format json -o profile.json
```

Output file:
```json
{
  "file": "data.csv",
  "row_count": 10000,
  "column_count": 8,
  "memory_usage_mb": 2.4,
  "columns": [
    {
      "name": "id",
      "dtype": "Int64",
      "null_count": 0,
      "null_ratio": 0.0,
      "unique_count": 10000,
      "unique_ratio": 1.0,
      "mean": 5000.5,
      "std": 2886.9,
      "min": 1,
      "max": 10000,
      "q25": 2500,
      "median": 5000,
      "q75": 7500
    },
    {
      "name": "age",
      "dtype": "Int64",
      "null_count": 500,
      "null_ratio": 0.05,
      "unique_count": 68,
      "unique_ratio": 0.0068,
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
  ],
  "patterns": {
    "email": ["email"],
    "join_date": ["date"]
  }
}
```

### Save to File

```bash
truthound profile data.parquet -o profile.json --format json
```

## Profile Statistics

### Basic Metrics

| Metric | Description |
|--------|-------------|
| `row_count` | Total number of rows |
| `column_count` | Total number of columns |
| `memory_usage_mb` | Estimated memory usage |

### Column Metrics

| Metric | Description | Types |
|--------|-------------|-------|
| `dtype` | Data type | All |
| `null_count` | Number of null values | All |
| `null_ratio` | Proportion of nulls | All |
| `unique_count` | Number of unique values | All |
| `unique_ratio` | Proportion of unique values | All |

### Numeric Metrics

| Metric | Description |
|--------|-------------|
| `mean` | Arithmetic mean |
| `std` | Standard deviation |
| `min` | Minimum value |
| `max` | Maximum value |
| `q25` | 25th percentile |
| `median` | 50th percentile (median) |
| `q75` | 75th percentile |
| `skewness` | Distribution asymmetry |
| `kurtosis` | Distribution tailedness |
| `entropy` | Information entropy |

### String Metrics

| Metric | Description |
|--------|-------------|
| `min_length` | Minimum string length |
| `max_length` | Maximum string length |
| `avg_length` | Average string length |
| `patterns` | Detected data patterns |

## Use Cases

### 1. Data Exploration

Quickly understand a new dataset:

```bash
truthound profile unknown_data.csv
```

### 2. Data Quality Baseline

Establish baseline statistics for monitoring:

```bash
truthound profile baseline.csv --format json -o baseline_profile.json
```

### 3. Documentation Generation

Generate profile for data catalog:

```bash
truthound profile production_table.parquet -o profile.json --format json
truthound docs generate profile.json -o data_docs.html
```

### 4. Pre-Analysis Check

Verify data before analysis:

```bash
# Check for data issues
truthound profile analysis_data.csv

# Look for:
# - High null ratios
# - Unexpected unique counts
# - Outlier min/max values
```

## Comparison with auto-profile

| Feature | `profile` | `auto-profile` |
|---------|-----------|----------------|
| Basic statistics | Yes | Yes |
| Pattern detection | Basic | Advanced |
| Correlation analysis | No | Optional |
| Rule generation | No | Yes |
| Sampling | No | Optional |
| Output formats | console, json | console, json, yaml |

For more advanced profiling, see [`auto-profile`](../profiler/auto-profile.md).

## Performance

The `profile` command is optimized for performance:

- Uses Polars LazyFrame for memory efficiency
- Single-pass computation where possible
- Efficient aggregation expressions

For very large files, consider using sampling:

```bash
# Use auto-profile with sampling for large datasets
truthound auto-profile large_data.parquet --sample 100000
```

## Related Commands

- [`auto-profile`](../profiler/auto-profile.md) - Advanced profiling with more options
- [`check`](check.md) - Validate data quality
- [`compare`](compare.md) - Compare datasets for drift

## See Also

- [Python API: th.profile()](../../python-api/core-functions.md#thprofile)
- [Data Profiling Tutorial](../../tutorials/data-profiling.md)
- [Data Docs Generation](../docs/generate.md)
