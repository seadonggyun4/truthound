# truthound benchmark compare

Compare two benchmark results to detect performance regressions or improvements.

## Synopsis

```bash
truthound benchmark compare <baseline> <current> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `baseline` | Yes | Path to baseline benchmark results (JSON format required) |
| `current` | Yes | Path to current benchmark results (JSON format required) |

!!! note "JSON Format Required"
    Both input files must be JSON benchmark results from `benchmark run`.
    The command will warn you if the file extension is not `.json`.

    ```bash
    # Generate benchmark results (JSON format auto-detected from .json extension)
    truthound benchmark run --suite ci -o results.json
    ```

!!! warning "Common Mistake"
    This command compares **benchmark result files**, not data files.
    If you pass data files (CSV, Parquet, etc.), you'll see a helpful error:

    ```
    Warning: baseline file 'data.csv' does not have .json extension.
    This command compares benchmark result JSON files, not data files.

    Did you mean to run a benchmark first?
      truthound benchmark run --suite ci -o baseline.json
    ```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--threshold` | `-t` | `10.0` | Performance difference threshold (%) |
| `--format` | `-f` | `console` | Output format (console, json) |

## Description

The `benchmark compare` command compares benchmark results:

1. **Loads** baseline and current benchmark results
2. **Compares** metrics for matching benchmarks
3. **Calculates** percentage changes
4. **Reports** regressions and improvements
5. **Returns** exit code 1 if regression detected

## Examples

### Basic Comparison

```bash
truthound benchmark compare baseline.json current.json
```

Output:
```
Benchmark Comparison
====================
Baseline: baseline.json (2025-01-14)
Current:  current.json (2025-01-15)
Threshold: 10.0%

Results
───────────────────────────────────────────────────────────────────
Benchmark       Baseline    Current     Change      Status
───────────────────────────────────────────────────────────────────
profile         0.376s      0.382s      +1.6%       OK
check           0.524s      0.498s      -5.0%       IMPROVED
scan            0.245s      0.289s      +18.0%      REGRESSION
compare         0.412s      0.425s      +3.2%       OK
───────────────────────────────────────────────────────────────────

Summary
───────────────────────────────────────────────────────────────────
Total Benchmarks: 4
Improved: 1 (check: -5.0%)
Regressed: 1 (scan: +18.0%)
Unchanged: 2

Status: REGRESSION DETECTED
The following benchmarks exceeded the 10.0% threshold:
  - scan: +18.0% (0.245s → 0.289s)
───────────────────────────────────────────────────────────────────
```

### Custom Threshold

```bash
# Stricter threshold (5%)
truthound benchmark compare baseline.json current.json --threshold 5.0

# More lenient threshold (20%)
truthound benchmark compare baseline.json current.json --threshold 20.0
```

### JSON Output

```bash
truthound benchmark compare baseline.json current.json --format json
```

Output:
```json
{
  "baseline_file": "baseline.json",
  "current_file": "current.json",
  "threshold_percent": 10.0,
  "regressions": [
    {
      "name": "scan",
      "baseline_seconds": 0.245,
      "current_seconds": 0.289,
      "change_percent": 18.0
    }
  ],
  "improvements": [
    {
      "name": "check",
      "baseline_seconds": 0.524,
      "current_seconds": 0.498,
      "change_percent": -5.0
    }
  ],
  "has_regressions": true
}
```

## Comparison Status

| Status | Condition | Description |
|--------|-----------|-------------|
| `OK` | Change < threshold | Within acceptable range |
| `IMPROVED` | Change < 0 | Performance improved |
| `REGRESSION` | Change > threshold | Performance degraded |

## Use Cases

### 1. CI/CD Regression Detection

```yaml
# GitHub Actions
- name: Run Benchmarks
  run: truthound benchmark run --suite ci -o current.json --format json

- name: Check for Regression
  run: |
    truthound benchmark compare \
      benchmarks/baseline.json \
      current.json \
      --threshold 10.0
```

### 2. Before/After Optimization

```bash
# Before optimization
truthound benchmark run --suite full -o before.json --format json

# ... make code changes ...

# After optimization
truthound benchmark run --suite full -o after.json --format json

# Compare
truthound benchmark compare before.json after.json
```

### 3. Release Validation

```bash
# Compare against last release
truthound benchmark compare \
  releases/v1.0.0_benchmark.json \
  releases/v1.1.0_benchmark.json \
  --threshold 5.0
```

### 4. Daily Performance Tracking

```bash
#!/bin/bash
# daily_benchmark.sh
TODAY=$(date +%Y%m%d)
YESTERDAY=$(date -d "yesterday" +%Y%m%d)

# Run today's benchmark
truthound benchmark run --suite ci -o "benchmarks/${TODAY}.json" --format json

# Compare with yesterday
if [ -f "benchmarks/${YESTERDAY}.json" ]; then
  truthound benchmark compare \
    "benchmarks/${YESTERDAY}.json" \
    "benchmarks/${TODAY}.json" \
    --threshold 15.0
fi
```

## Comparison Algorithm

1. **Match benchmarks** by name between baseline and current
2. **Calculate change**: `((current - baseline) / baseline) * 100`
3. **Apply threshold**: Mark as regression if change >= threshold
4. **Report status**: OK, IMPROVED, or REGRESSION

## Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 5% | Strict, production releases |
| 10% | Standard, CI/CD pipelines |
| 15% | Development, feature branches |
| 20% | Lenient, early development |

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (no regressions detected) |
| 1 | Error (regression detected, invalid arguments, or file not found) |

!!! note "CI/CD Integration"
    Exit code 1 on regression makes this command ideal for CI/CD pipelines.
    The build will fail if performance degrades beyond the threshold.

## Related Commands

- [`benchmark run`](run.md) - Run benchmarks
- [`benchmark list`](list.md) - List available benchmarks

## See Also

- [Benchmark Overview](index.md)
- [CI/CD Integration](../../guides/ci-cd.md)
