# truthound benchmark run

Run performance benchmarks.

## Synopsis

```bash
truthound benchmark run [BENCHMARK] [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `benchmark` | No | Specific benchmark to run (e.g., profile, check, scan) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--suite` | `-s` | None | Benchmark suite (quick, ci, full, profiling, validation) |
| `--size` | | `small` | Data size preset (tiny, small, medium, large, xlarge) |
| `--rows` | `-r` | None | Number of rows (overrides --size) |
| `--iterations` | `-i` | `3` | Number of iterations |
| `--warmup` | `-w` | `1` | Warmup iterations |
| `--output` | `-o` | None | Output file path |
| `--format` | `-f` | Auto | Output format (json, html). Auto-detected from `-o` file extension |
| `--save-baseline` | | `false` | Save results as baseline |
| `--compare-baseline` | | `false` | Compare with existing baseline |
| `--verbose` | `-v` | `false` | Verbose logging |

## Description

The `benchmark run` command executes performance benchmarks:

1. **Prepares** test data based on size preset
2. **Warms up** with initial iterations
3. **Executes** benchmark iterations
4. **Collects** timing and memory metrics
5. **Reports** results and statistics

## Benchmark Suites

| Suite | Estimated Time | Description | Benchmarks Included |
|-------|---------------|-------------|---------------------|
| `quick` | ~5 seconds | Fast verification | profile, check, learn (1K rows) |
| `ci` | ~15 seconds | CI/CD optimized | profile, check, learn, compare, scan (10K rows) |
| `full` | ~30 seconds | Core benchmarks | profile, check, learn, compare, scan, throughput (10K rows) |
| `profiling` | ~10 seconds | Profiling focused | All profiling category benchmarks |
| `validation` | ~10 seconds | Validation focused | All validation category benchmarks |

## Data Size Presets

| Size | Rows | Memory | Use Case |
|------|------|--------|----------|
| `tiny` | ~1,000 | < 10 MB | Quick tests |
| `small` | ~10,000 | < 50 MB | Development |
| `medium` | ~100,000 | ~200 MB | Default |
| `large` | ~1,000,000 | ~1 GB | Performance testing |
| `xlarge` | ~10,000,000 | ~5 GB | Stress testing |

## Examples

### Run Specific Benchmark

```bash
truthound benchmark run profile --size small
```

Output:
```
======================================================================
  BENCHMARK: single:profile
======================================================================

Environment: Python 3.13 on Darwin
Polars: 1.x, Truthound: 1.x

Results: 1/1 passed (100%)
Total Duration: 142.58ms

  [PROFILING]
    ✓ profile: 0.72ms (13.84M rows/s)

======================================================================
```

### Run Benchmark Suite

```bash
# Quick suite for fast verification (~5 seconds)
truthound benchmark run --suite quick

# CI/CD optimized suite (~15 seconds)
truthound benchmark run --suite ci

# Full benchmark suite (~30 seconds)
truthound benchmark run --suite full
```

### Custom Row Count

```bash
# Override size preset with exact row count
truthound benchmark run check --rows 1000000
```

### Custom Iterations

```bash
# More iterations for accuracy
truthound benchmark run profile --iterations 10 --warmup 2
```

### Save Results

```bash
# JSON output (auto-detected from .json extension)
truthound benchmark run --suite ci -o results.json

# HTML report (auto-detected from .html extension)
truthound benchmark run --suite full -o report.html

# Explicit format (overrides auto-detection)
truthound benchmark run --suite ci -o results.dat --format json
```

!!! note "Auto-detected Format"
    The output format is automatically detected from the file extension:

    - `.json` → JSON format (required for `benchmark compare`)
    - `.html` / `.htm` → HTML format (requires `pip install truthound[reports]`)
    - Other extensions → JSON format (default for file output)

!!! warning "HTML Report Dependency"
    HTML reports require Jinja2. Install with:
    ```bash
    pip install truthound[reports]
    ```

Output file (`results.json`):
```json
{
  "suite_name": "ci",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:15Z",
  "environment": {
    "python_version": "3.13.1",
    "polars_version": "1.x",
    "truthound_version": "1.x",
    "platform": "Darwin",
    "architecture": "arm64",
    "cpu_count": 10,
    "memory_total_gb": 36.0
  },
  "results": [
    {
      "benchmark_name": "profile",
      "benchmark_category": "profiling",
      "parameters": {
        "row_count": 10000,
        "iterations": 3
      },
      "metrics": {
        "timing": {
          "mean_seconds": 0.00072,
          "std_dev_seconds": 0.00011,
          "min_seconds": 0.00060,
          "max_seconds": 0.00090
        },
        "throughput": {
          "rows_per_second": 13840000
        }
      },
      "status": "passed"
    }
  ]
}
```

### Baseline Management

```bash
# Save results as baseline
truthound benchmark run --suite ci --save-baseline
```

Output:
```
Baseline saved to: .benchmarks/baseline.json
```

```bash
# Compare against baseline
truthound benchmark run --suite ci --compare-baseline
```

Output:
```
Benchmark Comparison
====================
Baseline: 2025-01-14 (ci suite)
Current:  2025-01-15 (ci suite)

Benchmark    Baseline    Current     Change      Status
───────────────────────────────────────────────────────────────────
profile      0.376s      0.382s      +1.6%       OK
check        0.524s      0.498s      -5.0%       IMPROVED
scan         0.245s      0.289s      +18.0%      REGRESSION
───────────────────────────────────────────────────────────────────

Status: REGRESSION DETECTED (1 benchmark)
Exit code: 1
```

### Verbose Output

```bash
truthound benchmark run profile --verbose
```

Shows detailed timing for each operation within the benchmark.

## Use Cases

### 1. Development Testing

```bash
# Quick check during development
truthound benchmark run profile --size tiny --iterations 3
```

### 2. Performance Optimization

```bash
# Detailed benchmark before optimization
truthound benchmark run --suite full -o before.json --format json

# ... make changes ...

# Compare after optimization
truthound benchmark compare before.json after.json
```

### 3. CI/CD Integration

```yaml
# GitHub Actions
- name: Performance Benchmark
  run: |
    truthound benchmark run --suite ci \
      --compare-baseline \
      --threshold 15.0
```

### 4. Size Scaling Analysis

```bash
# Test performance at different scales
for size in tiny small medium large; do
  truthound benchmark run profile \
    --size $size \
    -o "profile_${size}.json" \
    --format json
done
```

### 5. Stress Testing

```bash
# Large-scale stress test
truthound benchmark run --suite full \
  --size xlarge \
  --iterations 3 \
  -o stress_test.html \
  --format html
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (regression detected with --compare-baseline, invalid arguments, or benchmark error) |

## Related Commands

- [`benchmark list`](list.md) - List available benchmarks
- [`benchmark compare`](compare.md) - Compare benchmark results

## See Also

- [Benchmark Overview](index.md)
- [Performance Guide](../../guides/performance.md)
