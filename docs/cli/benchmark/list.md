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
Available Benchmarks
====================

Profiling
─────────────────────────────────────────────────────────────────
profile          Basic data profiling
auto-profile     Advanced profiling with pattern detection
quick-suite      Profile and generate rules

Validation
─────────────────────────────────────────────────────────────────
check            Data quality validation
scan             PII scanning
mask             Data masking

Comparison
─────────────────────────────────────────────────────────────────
compare          Dataset drift comparison
ml-drift         ML-based drift detection

Schema
─────────────────────────────────────────────────────────────────
learn            Schema inference
schema-validate  Schema validation

I/O
─────────────────────────────────────────────────────────────────
read-csv         CSV reading performance
read-parquet     Parquet reading performance
write-csv        CSV writing performance
write-parquet    Parquet writing performance

─────────────────────────────────────────────────────────────────
Total: 13 benchmarks

Suites:
  quick:      profile, check
  ci:         profile, check, scan, compare
  full:       All benchmarks
  profiling:  profile, auto-profile, quick-suite
  validation: check, scan, mask

Run with: truthound benchmark run <benchmark> [OPTIONS]
```

### JSON Output

```bash
truthound benchmark list --format json
```

Output:
```json
{
  "benchmarks": {
    "profiling": [
      {
        "name": "profile",
        "description": "Basic data profiling",
        "suites": ["quick", "ci", "full", "profiling"]
      },
      {
        "name": "auto-profile",
        "description": "Advanced profiling with pattern detection",
        "suites": ["full", "profiling"]
      },
      {
        "name": "quick-suite",
        "description": "Profile and generate rules",
        "suites": ["full", "profiling"]
      }
    ],
    "validation": [
      {
        "name": "check",
        "description": "Data quality validation",
        "suites": ["quick", "ci", "full", "validation"]
      },
      {
        "name": "scan",
        "description": "PII scanning",
        "suites": ["ci", "full", "validation"]
      },
      {
        "name": "mask",
        "description": "Data masking",
        "suites": ["full", "validation"]
      }
    ],
    "comparison": [
      {
        "name": "compare",
        "description": "Dataset drift comparison",
        "suites": ["ci", "full"]
      },
      {
        "name": "ml-drift",
        "description": "ML-based drift detection",
        "suites": ["full"]
      }
    ],
    "schema": [
      {
        "name": "learn",
        "description": "Schema inference",
        "suites": ["full"]
      },
      {
        "name": "schema-validate",
        "description": "Schema validation",
        "suites": ["full"]
      }
    ],
    "io": [
      {
        "name": "read-csv",
        "description": "CSV reading performance",
        "suites": ["full"]
      },
      {
        "name": "read-parquet",
        "description": "Parquet reading performance",
        "suites": ["full"]
      },
      {
        "name": "write-csv",
        "description": "CSV writing performance",
        "suites": ["full"]
      },
      {
        "name": "write-parquet",
        "description": "Parquet writing performance",
        "suites": ["full"]
      }
    ]
  },
  "suites": {
    "quick": ["profile", "check"],
    "ci": ["profile", "check", "scan", "compare"],
    "full": ["all"],
    "profiling": ["profile", "auto-profile", "quick-suite"],
    "validation": ["check", "scan", "mask"]
  },
  "total": 13
}
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

### 2. Filter by Suite

```bash
# Find benchmarks in CI suite
truthound benchmark list --format json | jq '.suites.ci'
```

### 3. Script Integration

```bash
# Get benchmark names for scripting
truthound benchmark list --format json | jq -r '.benchmarks | keys[]'
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
