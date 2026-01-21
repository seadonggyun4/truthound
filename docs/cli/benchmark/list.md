# truthound benchmark list

List available benchmarks by category.

## Synopsis

```bash
truthound benchmark list [OPTIONS]
```

## Arguments

None.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `console` | Output format (console, json) |

## Description

The `benchmark list` command displays all available benchmarks:

1. **Lists** benchmarks by category
2. **Shows** benchmark descriptions
3. **Indicates** included suites

## Examples

### Console Output

```bash
truthound benchmark list
```

Output:
```
Available Benchmarks:
============================================================

[PROFILING]
  profile              - Basic data profiling
  auto-profile         - Advanced profiling with pattern detection
  quick-suite          - Profile and generate rules

[VALIDATION]
  check                - Data quality validation
  scan                 - PII scanning
  mask                 - Data masking

[COMPARISON]
  compare              - Dataset drift comparison
  ml-drift             - ML-based drift detection
```

### JSON Output

```bash
truthound benchmark list --format json
```

Output:
```json
[
  {
    "name": "profile",
    "category": "profiling",
    "description": "Basic data profiling"
  },
  {
    "name": "check",
    "category": "validation",
    "description": "Data quality validation"
  },
  {
    "name": "compare",
    "category": "comparison",
    "description": "Dataset drift comparison"
  }
]
```

## Benchmark Categories

### Profiling

| Benchmark | Description |
|-----------|-------------|
| `profile` | Basic data profiling |
| `auto-profile` | Advanced profiling with patterns |
| `quick-suite` | Profile and rule generation |

### Validation

| Benchmark | Description |
|-----------|-------------|
| `check` | Data quality validation |
| `scan` | PII scanning |
| `mask` | Data masking |

### Comparison

| Benchmark | Description |
|-----------|-------------|
| `compare` | Dataset drift comparison |
| `ml-drift` | ML-based drift detection |

### Schema

| Benchmark | Description |
|-----------|-------------|
| `learn` | Schema inference |
| `schema-validate` | Schema validation |

### I/O

| Benchmark | Description |
|-----------|-------------|
| `read-csv` | CSV reading performance |
| `read-parquet` | Parquet reading performance |
| `write-csv` | CSV writing performance |
| `write-parquet` | Parquet writing performance |

## Suite Contents

| Suite | Benchmarks |
|-------|------------|
| `quick` | profile, check |
| `ci` | profile, check, scan, compare |
| `full` | All benchmarks |
| `profiling` | profile, auto-profile, quick-suite |
| `validation` | check, scan, mask |

## Use Cases

### 1. Discover Benchmarks

```bash
# See what benchmarks are available
truthound benchmark list
```

### 2. Filter by Category

```bash
# Find benchmarks in profiling category
truthound benchmark list --format json | jq '.[] | select(.category == "profiling")'
```

### 3. Script Integration

```bash
# Get benchmark names for scripting
truthound benchmark list --format json | jq -r '.[].name'
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |

## Related Commands

- [`benchmark run`](run.md) - Run benchmarks
- [`benchmark compare`](compare.md) - Compare benchmark results

## See Also

- [Benchmark Overview](index.md)
