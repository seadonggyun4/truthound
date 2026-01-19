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
| `--size` | | `medium` | Data size preset (tiny, small, medium, large, xlarge) |
| `--rows` | `-r` | None | Number of rows (overrides --size) |
| `--iterations` | `-i` | `5` | Number of iterations |
| `--warmup` | `-w` | `2` | Warmup iterations |
| `--output` | `-o` | None | Output file path |
| `--format` | `-f` | `console` | Output format (console, json, html) |
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

| Suite | Description | Benchmarks Included |
|-------|-------------|---------------------|
| `quick` | Fast verification | Core operations only |
| `ci` | CI/CD optimized | Balanced coverage |
| `full` | Complete suite | All benchmarks |
| `profiling` | Profiling focused | Profile, auto-profile |
| `validation` | Validation focused | Check, scan, validators |

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
truthound benchmark run profile --size medium
```

Output:
```
Benchmark: profile
==================
Size: medium (100,000 rows)
Iterations: 5
Warmup: 2

Warmup 1/2... done (0.45s)
Warmup 2/2... done (0.42s)

Iteration 1/5: 0.38s
Iteration 2/5: 0.37s
Iteration 3/5: 0.39s
Iteration 4/5: 0.36s
Iteration 5/5: 0.38s

Results
───────────────────────────────────────────────────────────────────
Mean:       0.376s
Std Dev:    0.011s
Min:        0.360s
Max:        0.390s
Throughput: 265,957 rows/s
Memory:     156 MB (peak)
───────────────────────────────────────────────────────────────────
```

### Run Benchmark Suite

```bash
# Quick suite for fast verification
truthound benchmark run --suite quick

# CI/CD optimized suite
truthound benchmark run --suite ci

# Full comprehensive suite
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
truthound benchmark run profile --iterations 10 --warmup 3
```

### Save Results

```bash
# JSON output
truthound benchmark run --suite ci -o results.json --format json

# HTML report
truthound benchmark run --suite full -o report.html --format html
```

Output file (`results.json`):
```json
{
  "suite": "ci",
  "timestamp": "2025-01-15T10:30:00Z",
  "system": {
    "cpu": "Intel Core i7-12700K",
    "memory": "32 GB",
    "os": "Ubuntu 22.04"
  },
  "benchmarks": [
    {
      "name": "profile",
      "size": "medium",
      "rows": 100000,
      "iterations": 5,
      "results": {
        "mean": 0.376,
        "std_dev": 0.011,
        "min": 0.360,
        "max": 0.390,
        "throughput": 265957,
        "memory_peak_mb": 156
      }
    },
    {
      "name": "check",
      "size": "medium",
      "rows": 100000,
      "iterations": 5,
      "results": {
        "mean": 0.524,
        "std_dev": 0.015,
        "min": 0.505,
        "max": 0.548,
        "throughput": 190839,
        "memory_peak_mb": 189
      }
    }
  ],
  "summary": {
    "total_benchmarks": 2,
    "total_time": 12.5,
    "status": "passed"
  }
}
```

### Baseline Management

```bash
# Save results as baseline
truthound benchmark run --suite ci --save-baseline
```

Output:
```
Baseline saved to: .truthound/benchmarks/baseline.json
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
| 0 | Success (all benchmarks passed) |
| 1 | Regression detected (with --compare-baseline) |
| 2 | Invalid arguments or benchmark error |

## Related Commands

- [`benchmark list`](list.md) - List available benchmarks
- [`benchmark compare`](compare.md) - Compare benchmark results

## See Also

- [Benchmark Overview](index.md)
- [Performance Guide](../../guides/performance.md)
