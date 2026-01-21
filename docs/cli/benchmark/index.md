# Benchmark Commands

Performance testing commands for measuring and comparing Truthound operations.

!!! note "HTML Report Dependency"
    HTML benchmark reports require Jinja2. Install with: `pip install truthound[reports]`

## Quick Start

```bash
# Run quick benchmark suite
truthound benchmark run --suite quick

# Run single 'profile' benchmark
truthound benchmark run profile

# List available benchmarks
truthound benchmark list

# Compare results
truthound benchmark compare baseline.json current.json
```

## Overview

| Command | Description | Primary Use Case |
|---------|-------------|------------------|
| [`run`](run.md) | Run performance benchmarks | Performance testing |
| [`list`](list.md) | List available benchmarks | Discovery |
| [`compare`](compare.md) | Compare benchmark results | Regression detection |

!!! tip "Common Mistake"
    `profile`, `check`, `scan` are **benchmark names**, not subcommands.
    Use `benchmark run profile`, not `benchmark profile`.

## What are Benchmarks?

Benchmarks measure the performance of Truthound operations:

- **Profiling benchmarks** - Measure data profiling speed
- **Validation benchmarks** - Measure validation throughput
- **I/O benchmarks** - Measure read/write performance
- **Regression detection** - Compare against baselines

## Benchmark Suites

| Suite | Estimated Time | Description | Use Case |
|-------|---------------|-------------|----------|
| `quick` | ~5 seconds | Fast verification (1K rows) | Quick checks |
| `ci` | ~15 seconds | CI/CD optimized (10K rows) | Automated pipelines |
| `full` | ~30 seconds | Core benchmarks (10K rows) | Comprehensive testing |
| `profiling` | ~10 seconds | Profiling-related benchmarks | Profile performance |
| `validation` | ~10 seconds | Validation-related benchmarks | Validator performance |

## Data Size Presets

| Size | Description | Approximate Rows |
|------|-------------|------------------|
| `tiny` | Very small dataset | ~1,000 |
| `small` | Small dataset (default) | ~10,000 |
| `medium` | Medium dataset | ~100,000 |
| `large` | Large dataset | ~1,000,000 |
| `xlarge` | Very large dataset | ~10,000,000 |

## Workflow

```mermaid
graph LR
    A[benchmark list] --> B[Select Benchmark]
    B --> C[benchmark run]
    C --> D[Results]
    C --> E[--save-baseline]
    E --> F[baseline.json]
    C --> G[--compare-baseline]
    G --> H[Regression Report]
    D --> I[benchmark compare]
    I --> H
```

## Quick Examples

### Run Benchmarks

```bash
# Run specific benchmark
truthound benchmark run profile --size small

# Run benchmark suite (~5 seconds)
truthound benchmark run --suite quick

# CI/CD suite (~15 seconds)
truthound benchmark run --suite ci

# Custom row count
truthound benchmark run check --rows 1000000
```

### List Benchmarks

```bash
# List all benchmarks
truthound benchmark list

# JSON output
truthound benchmark list --format json
```

### Compare Results

```bash
# Compare two benchmark results (JSON format auto-detected from .json extension)
truthound benchmark compare baseline.json current.json

# Custom threshold
truthound benchmark compare old.json new.json --threshold 5.0
```

## CI/CD Integration

### GitHub Actions

```yaml
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Benchmarks
        run: |
          truthound benchmark run --suite ci \
            -o benchmark_results.json \
            --format json

      - name: Compare with Baseline
        run: |
          truthound benchmark compare \
            benchmarks/baseline.json \
            benchmark_results.json \
            --threshold 10.0

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results.json
```

### Baseline Management

```bash
# Save new baseline
truthound benchmark run --suite ci --save-baseline

# Compare against saved baseline
truthound benchmark run --suite ci --compare-baseline
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| `execution_time` | Total execution time (seconds) |
| `throughput` | Records processed per second |
| `memory_peak` | Peak memory usage (MB) |
| `iterations` | Number of benchmark iterations |

## Use Cases

### 1. Quick Development Feedback

```bash
# Fast verification during development (~5 seconds)
truthound benchmark run --suite quick
```

### 2. CI/CD Performance Testing

```bash
# CI-optimized suite (~15 seconds)
# JSON format auto-detected from .json extension
truthound benchmark run --suite ci -o results.json
```

### 3. Regression Detection

```bash
# Before changes
truthound benchmark run --suite ci --save-baseline

# After changes
truthound benchmark run --suite ci --compare-baseline
```

### 4. Size Scaling Analysis

```bash
# Test different data sizes (JSON format auto-detected)
for size in tiny small medium large; do
  truthound benchmark run profile --size $size -o "results_${size}.json"
done
```

### 5. Comprehensive Testing

```bash
# Full suite for thorough testing (~30 seconds)
truthound benchmark run --suite full --iterations 5
```

## Command Reference

- [run](run.md) - Run performance benchmarks
- [list](list.md) - List available benchmarks
- [compare](compare.md) - Compare benchmark results

## See Also

- [Performance Guide](../../guides/performance.md)
- [CI/CD Integration](../../guides/ci-cd.md)
