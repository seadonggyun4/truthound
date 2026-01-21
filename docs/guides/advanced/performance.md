# Performance Guide

This document provides comprehensive performance characteristics, benchmarks, and optimization guidelines for Truthound.

---

## Table of Contents

1. [Performance Overview](#1-performance-overview)
2. [Time Complexity Analysis](#2-time-complexity-analysis)
3. [Memory Complexity Analysis](#3-memory-complexity-analysis)
4. [Benchmarks](#4-benchmarks)
5. [Optimization Strategies](#5-optimization-strategies)
6. [Large Dataset Handling](#6-large-dataset-handling)
7. [Streaming Validation](#7-streaming-validation)
8. [Best Practices](#8-best-practices)
9. [Internal Performance Optimizations](#9-internal-performance-optimizations)

---

## 1. Performance Overview

### Design Philosophy

Truthound is built on Polars, which provides:
- **Lazy Evaluation**: Query plans are optimized before execution
- **Columnar Storage**: Efficient memory layout for analytical operations
- **SIMD Operations**: Vectorized processing for modern CPUs
- **Multi-threading**: Parallel execution across CPU cores

### Key Performance Characteristics

| Aspect | Description |
|--------|-------------|
| **Data Format** | Polars LazyFrame (lazy evaluation) |
| **Processing** | Columnar, vectorized operations |
| **Parallelism** | Multi-threaded by default |
| **Memory** | Efficient columnar memory layout |
| **I/O** | Streaming support for large files |

---

## 2. Time Complexity Analysis

### Validator Categories

#### O(n) - Linear Complexity

Most validators operate in linear time relative to row count:

| Validator Type | Time Complexity | Notes |
|----------------|-----------------|-------|
| Null Check | O(n) | Single column scan |
| Range Validation | O(n) | Single column scan |
| Pattern Matching | O(n * m) | n rows, m pattern length |
| Type Check | O(n) | Single column scan |
| Completeness | O(n) | Aggregate operations |

#### O(n log n) - Linearithmic Complexity

Validators requiring sorting or ordering:

| Validator Type | Time Complexity | Notes |
|----------------|-----------------|-------|
| Uniqueness | O(n log n) | Hash-based deduplication |
| Percentile | O(n log n) | Partial sorting |
| Median | O(n log n) | Partial sorting |
| Time Series Order | O(n log n) | Sorting required |

#### O(n * m) - Multi-column Operations

Cross-column or cross-table validators:

| Validator Type | Time Complexity | Notes |
|----------------|-----------------|-------|
| Foreign Key | O(n + m) | n child rows, m parent rows |
| Column Comparison | O(n * k) | n rows, k columns |
| Multi-column Pattern | O(n * k) | n rows, k columns |

#### O(n^2) - Quadratic Complexity (Avoided)

These operations use sampling to avoid quadratic complexity:

| Validator Type | Native Complexity | With Sampling |
|----------------|-------------------|---------------|
| Pairwise Correlation | O(n^2) | O(sample_size * k) |
| Duplicate Detection | O(n^2) | O(n log n) using hash |
| Anomaly Detection | O(n^2) | O(n * sample_size) |

### Statistical Validators

| Validator | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| KS Test (Drift) | O(n log n) | O(n) |
| Chi-Square | O(n + k) | O(k) categories |
| PSI | O(n + b) | O(b) buckets |
| IQR Outlier | O(n) | O(1) |
| Z-Score | O(n) | O(1) |
| Isolation Forest | O(n * t * log(s)) | O(t * s) |

Where:
- n = number of rows
- k = number of categories/columns
- b = number of buckets
- t = number of trees
- s = sample size per tree

---

## 3. Memory Complexity Analysis

### Base Memory Usage

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| LazyFrame Reference | O(1) | Metadata only |
| Collected DataFrame | O(n * k * w) | n rows, k columns, w bytes/value |
| Validation Issues | O(i) | i = number of issues |

### Validator Memory Patterns

#### Low Memory Validators (Streaming Compatible)

| Validator | Memory Usage | Notes |
|-----------|--------------|-------|
| Null Check | O(1) | Aggregate result only |
| Range Check | O(1) | Min/max tracking |
| Row Count | O(1) | Counter only |
| Type Check | O(k) | k distinct types |

#### Medium Memory Validators

| Validator | Memory Usage | Notes |
|-----------|--------------|-------|
| Uniqueness | O(d) | d = distinct values |
| Pattern Match | O(m) | m = pattern length |
| Value Set | O(s) | s = allowed values |
| Statistics | O(k) | k = statistical measures |

#### High Memory Validators

| Validator | Memory Usage | Notes |
|-----------|--------------|-------|
| Anomaly Detection | O(n * f) | n samples, f features |
| Correlation Matrix | O(k^2) | k columns |
| Cross-Table Join | O(min(n, m)) | Anti-join result |
| Profiling | O(n * k) | Full data scan |

### Memory Optimization Features

```python
# Sample-based validation for large datasets
from truthound.validators.referential import ForeignKeyValidator

validator = ForeignKeyValidator(
    child_table="orders",
    child_columns=["customer_id"],
    parent_table="customers",
    parent_columns=["id"],
    sample_size=100_000,  # Validate on 100k sample
    sample_seed=42,       # Reproducible results
)
```

---

## 4. Benchmarks

### Important Notice

The benchmark values in this section are illustrative examples based on the expected behavior of Polars-based operations. Actual performance depends on:

- Hardware specifications (CPU, RAM, storage)
- Data characteristics (column types, cardinality, null ratio)
- Polars version and configuration
- Operating system and Python version

**Users should run their own benchmarks** to establish baseline performance for their specific environment and workloads.

### Running Benchmarks with CLI

Truthound provides a built-in benchmark CLI for measuring performance:

```bash
# Quick benchmark (~5 seconds)
truthound benchmark run --suite quick

# CI/CD optimized benchmark (~15 seconds)
truthound benchmark run --suite ci

# Full benchmark suite (~30 seconds)
truthound benchmark run --suite full

# Single operation benchmark
truthound benchmark run profile --size medium --iterations 5
```

**Benchmark Suites:**

| Suite | Estimated Time | Data Size | Use Case |
|-------|---------------|-----------|----------|
| `quick` | ~5 seconds | 1K rows | Development feedback |
| `ci` | ~15 seconds | 10K rows | CI/CD pipelines |
| `full` | ~30 seconds | 10K rows | Comprehensive testing |

For detailed CLI options, see the [Benchmark Command Reference](../cli/benchmark/index.md).

### Running Custom Benchmarks

```python
import time
import polars as pl
from truthound.validators.completeness import NullValidator

# Load your data
lf = pl.scan_parquet("your_data.parquet")
row_count = lf.select(pl.count()).collect().item()

# Benchmark a validator
validator = NullValidator(column="your_column")
start = time.perf_counter()
issues = validator.validate(lf)
duration = time.perf_counter() - start

print(f"Rows: {row_count:,}")
print(f"Duration: {duration:.3f}s")
print(f"Throughput: {row_count / duration / 1_000_000:.2f}M rows/s")
```

### Performance Characteristics

The following describes general performance characteristics, not specific benchmarks:

| Operation Type | Complexity | Notes |
|----------------|------------|-------|
| Null/Range checks | O(n) | Linear scan, highly efficient |
| Uniqueness checks | O(n log n) | Hash-based deduplication |
| Cross-table joins | O(n + m) | Depends on join strategy |
| Streaming validation | O(n) | Bounded memory with chunking |
| ML-based detection | O(n * features) | Depends on algorithm |

---

## 5. Optimization Strategies

### 5.1 Lazy Evaluation

```python
import polars as pl
from truthound.validators.completeness import NullValidator

# Use LazyFrame for deferred execution
lf = pl.scan_parquet("large_file.parquet")

# Validator adds operations to query plan
validator = NullValidator(column="customer_id")

# Execution happens only when needed
issues = validator.validate(lf)
```

### 5.2 Column Projection

```python
# Only load required columns
lf = pl.scan_parquet(
    "large_file.parquet",
    columns=["customer_id", "order_date", "amount"]  # Explicit selection
)
```

### 5.3 Predicate Pushdown

```python
# Filter early in the pipeline
lf = pl.scan_parquet("orders.parquet").filter(
    pl.col("order_date") >= "2024-01-01"
)

# Validation runs on filtered data
issues = validator.validate(lf)
```

### 5.4 Sample-Based Validation

```python
from truthound.validators.referential import ForeignKeyValidator

# For large datasets, use sampling
validator = ForeignKeyValidator(
    child_table="transactions",
    child_columns=["account_id"],
    parent_table="accounts",
    parent_columns=["id"],
    sample_size=100_000,  # Validate representative sample
    sample_seed=42,       # Reproducibility
)

# Results include estimated total violations
issues = validator.validate(transactions_lf)
# Output: "Found 523 violations in sample of 100,000 rows (0.52%).
#          Estimated 52,300 total violations in 10,000,000 rows."
```

### 5.5 Parallel Execution

Truthound provides built-in DAG-based parallel execution that automatically handles dependencies:

```python
import truthound as th

# Simple parallel execution via API
report = th.check("data.csv", parallel=True)

# With custom worker count
report = th.check("data.csv", parallel=True, max_workers=8)
```

For advanced control:

```python
from truthound.validators.optimization import (
    ValidatorDAG,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
)

# Build execution plan
dag = ValidatorDAG()
dag.add_validators([
    NullValidator(column="id"),
    RangeValidator(column="amount", min_value=0),
    PatternValidator(column="email", pattern=r"^[\w.-]+@[\w.-]+\.\w+$"),
])

plan = dag.build_execution_plan()

# Execute with parallel strategy
strategy = ParallelExecutionStrategy(max_workers=4)
result = plan.execute(lf, strategy)

print(f"Total issues: {len(result.all_issues)}")
print(f"Execution time: {result.total_duration_ms:.2f}ms")
```

Manual parallel execution (legacy approach):

```python
from concurrent.futures import ThreadPoolExecutor
from truthound.validators import NullValidator, RangeValidator, PatternValidator

validators = [
    NullValidator(column="id"),
    RangeValidator(column="amount", min_value=0),
    PatternValidator(column="email", pattern=r"^[\w.-]+@[\w.-]+\.\w+$"),
]

def run_validator(v, lf):
    return v.validate(lf)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_validator, v, lf) for v in validators]
    all_issues = [f.result() for f in futures]
```

### 5.6 Enterprise-Scale Sampling

For 100M+ row datasets, use enterprise sampling strategies:

```python
from truthound.profiler.enterprise_sampling import (
    EnterpriseScaleSampler,
    EnterpriseScaleConfig,
    MemoryBudgetConfig,
)

# Configure for large dataset
config = EnterpriseScaleConfig(
    target_rows=100_000,
    memory_budget=MemoryBudgetConfig(max_memory_mb=512),
    time_budget_seconds=60.0,
)

sampler = EnterpriseScaleSampler(config)

# Auto-select best strategy based on data size
result = sampler.sample(lf)

# Or specify strategy explicitly
result = sampler.sample(lf, strategy="block")       # For 10M-100M rows
result = sampler.sample(lf, strategy="multi_stage") # For 100M+ rows
```

Available strategies:

| Strategy | Best For | Memory | Speed |
|----------|----------|--------|-------|
| `block` | 10M-100M rows | O(1) | Fast |
| `multi_stage` | 100M-1B rows | O(1) | Medium |
| `column_aware` | Mixed column types | O(1) | Medium |
| `progressive` | Unknown distributions | O(1) | Adaptive |

---

## 6. Large Dataset Handling

### 6.1 Memory-Efficient Processing

```python
from truthound.validators.streaming import (
    stream_validate,
    stream_validate_many,
    ParquetStreamingSource,
)

# Single validator streaming
validator = NullValidator(column="customer_id")
issues = stream_validate(
    validator,
    "huge_file.parquet",
    chunk_size=100_000,
)

# Multiple validators in single pass
validators = [
    NullValidator(column="customer_id"),
    RangeValidator(column="amount", min_value=0, max_value=1_000_000),
]
results = stream_validate_many(
    validators,
    "huge_file.parquet",
    chunk_size=100_000,
)
```

### 6.2 Cross-Table Validation at Scale

```python
from truthound.validators.referential import ForeignKeyValidator

# For billion-row tables, use sampling
validator = ForeignKeyValidator(
    child_table="events",
    child_columns=["user_id"],
    parent_table="users",
    parent_columns=["id"],
    sample_size=500_000,   # 500K sample from child table
    sample_seed=42,
)

# Parent table keys are loaded in full (usually smaller)
# Child table is sampled for memory efficiency
validator.register_table("events", events_lf)
validator.register_table("users", users_lf)
issues = validator.validate(events_lf)
```

### 6.3 Partitioned Data Processing

```python
import polars as pl
from pathlib import Path

# Process partitioned data efficiently
partition_dir = Path("data/events/")
all_issues = []

for partition_file in partition_dir.glob("year=*/month=*/*.parquet"):
    lf = pl.scan_parquet(partition_file)
    issues = validator.validate(lf)
    all_issues.extend(issues)
```

---

## 7. Streaming Validation

### 7.1 Basic Streaming

```python
from truthound.validators.streaming import (
    ParquetStreamingSource,
    CSVStreamingSource,
    StreamingValidatorAdapter,
)

# Wrap any validator for streaming
adapter = StreamingValidatorAdapter(
    NullValidator(column="id"),
    chunk_size=100_000,
)

# Stream through file
issues = adapter.validate_streaming("huge_file.parquet")
```

### 7.2 Progress Monitoring

```python
def progress_callback(chunk_idx: int, total_chunks: int):
    pct = (chunk_idx + 1) / total_chunks * 100 if total_chunks > 0 else 0
    print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({pct:.1f}%)")

issues = adapter.validate_streaming(
    "huge_file.parquet",
    on_chunk=progress_callback,
)
```

### 7.3 Early Termination

```python
# Use iterator for early termination on critical issues
for chunk_idx, chunk_issues in adapter.validate_streaming_iter("data.parquet"):
    critical_issues = [i for i in chunk_issues if i.severity == "critical"]
    if len(critical_issues) > 100:
        print(f"Too many critical issues at chunk {chunk_idx}, stopping.")
        break
```

### 7.4 Custom Accumulators

```python
from truthound.validators.streaming import (
    StreamingAccumulator,
    CountingAccumulator,
    SamplingAccumulator,
)

# Count issues across chunks
counting_acc = CountingAccumulator()

# Keep sample of issues (max 100)
sampling_acc = SamplingAccumulator(max_samples=100)

issues = adapter.validate_streaming(
    "data.parquet",
    accumulator=sampling_acc,  # Only keep 100 sample issues
)
```

---

## 8. Best Practices

### 8.1 Data Loading

| Practice | Recommendation |
|----------|----------------|
| File Format | Prefer Parquet over CSV for large files |
| Column Selection | Only load columns needed for validation |
| Filtering | Apply filters early using LazyFrame |
| Streaming | Use streaming for files > 10GB |

### 8.2 Validator Selection

| Scenario | Recommendation |
|----------|----------------|
| Simple checks | Use specific validators (Null, Range) |
| Complex rules | Use QueryValidator with SQL |
| Large tables | Enable sampling for cross-table validators |
| Real-time | Use streaming validators |

### 8.3 Memory Management

| Practice | Recommendation |
|----------|----------------|
| LazyFrame | Always prefer LazyFrame over DataFrame |
| Chunk Size | 100K rows per chunk for streaming |
| Sample Size | 100K-500K for cross-table validation |
| Profiling | Use sample-based profiling for > 1M rows |

### 8.4 Performance Monitoring

```python
import time
from truthound.validators.base import ValidationIssue

class TimedValidator:
    """Wrapper to measure validator performance."""

    def __init__(self, validator):
        self.validator = validator
        self.last_duration = 0.0

    def validate(self, lf):
        start = time.perf_counter()
        issues = self.validator.validate(lf)
        self.last_duration = time.perf_counter() - start
        return issues

    def get_stats(self) -> dict:
        return {
            "validator": self.validator.name,
            "duration_ms": self.last_duration * 1000,
        }
```

### 8.5 SQL Query Security

```python
from truthound.validators.query import QueryValidator

# SQL queries are validated for security by default
validator = QueryValidator(
    query="SELECT * FROM data WHERE amount > 1000",
    validate_sql=True,          # Enable SQL injection protection
    allowed_tables=["data"],     # Whitelist allowed tables
)

# Dangerous patterns are blocked:
# - DDL statements (CREATE, DROP, ALTER)
# - DCL statements (GRANT, REVOKE)
# - Multiple statements (;SELECT...)
# - SQL injection patterns (UNION ALL SELECT)
```

---

## 9. Internal Performance Optimizations

Truthound implements several internal optimizations for maximum performance. These are automatically applied and don't require user configuration.

### 9.1 Expression-Based Validator Architecture

Validators that support expression-based execution can be batched into a single `collect()` call.

**Implementation**: `src/truthound/validators/base.py`

| Component | Description |
|-----------|-------------|
| `ValidationExpressionSpec` | Defines validation expression spec (column, type, count_expr, non_null_expr) |
| `ExpressionValidatorMixin` | Mixin for single-validator expression-based execution |
| `ExpressionBatchExecutor` | Batches multiple validators into single collect() |

```python
from truthound.validators.base import ExpressionBatchExecutor
from truthound.validators.completeness.null import NullValidator
from truthound.validators.distribution.range import RangeValidator

# Batch execution (single collect())
executor = ExpressionBatchExecutor()
executor.add_validator(NullValidator())
executor.add_validator(RangeValidator(min_value=0))
all_issues = executor.execute(lf)  # Single collect() for all validators
```

**Supported validators**:
- `NullValidator`, `NotNullValidator`, `CompletenessRatioValidator` (completeness)
- `BetweenValidator`, `RangeValidator`, `PositiveValidator`, `NonNegativeValidator` (range)

### 9.2 Lazy Loading Validator Registry

The validator registry uses lazy loading to minimize startup time.

**Implementation**: `src/truthound/validators/_lazy.py`

```python
# 200+ validators mapped to their modules
VALIDATOR_IMPORT_MAP: dict[str, str] = {
    "NullValidator": "truthound.validators.completeness.null",
    "BetweenValidator": "truthound.validators.distribution.range",
    # ... 200+ validators
}

# Category-based lazy loading
CATEGORY_MODULES: dict[str, str] = {
    "completeness": "truthound.validators.completeness",
    "distribution": "truthound.validators.distribution",
    # ... 28 categories
}
```

**Metrics tracking**: `ValidatorImportMetrics` class tracks import success/failure counts and timing.

### 9.3 xxhash Cache Optimization

Cache fingerprinting uses xxhash when available for ~10x faster hashing.

**Implementation**: `src/truthound/cache.py`

```python
try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

def _fast_hash(content: str) -> str:
    if _HAS_XXHASH:
        return xxhash.xxh64(content.encode()).hexdigest()[:16]
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

To enable: `pip install xxhash`

### 9.4 Native Polars Expressions (No map_elements)

All masking operations use native Polars expressions instead of Python callbacks.

**Implementation**: `src/truthound/maskers.py`

| Function | Optimization |
|----------|-------------|
| `_apply_redact()` | `pl.when/then/otherwise` chains, `str.replace_all()` |
| `_apply_hash()` | Polars native `hash()` (xxhash3 internally) |
| `_apply_fake()` | Hash-based deterministic generation |

```python
# Hash masking - no Python callbacks
def _apply_hash(df: pl.DataFrame, col: str) -> pl.DataFrame:
    c = pl.col(col)
    hashed = c.hash().cast(pl.String).str.slice(0, 16)
    return df.with_columns(
        pl.when(c.is_null()).then(pl.lit(None)).otherwise(hashed).alias(col)
    )
```

**Streaming mode**: Large datasets (>1M rows) use streaming engine:
```python
df = lf.collect(engine="streaming")
```

### 9.5 Heap-Based Report Sorting

Validation reports use heap-based sorting for O(1) most-severe-issue access.

**Implementation**: `src/truthound/report.py`

```python
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

def add_issue(self, issue: ValidationIssue) -> None:
    self.issues.append(issue)
    heapq.heappush(
        self._issues_heap,
        (_SEVERITY_ORDER[issue.severity], self._heap_counter, issue),
    )

def add_issues(self, issues: Iterable[ValidationIssue]) -> None:
    for issue in issues:
        self.issues.append(issue)
        self._issues_heap.append(...)
        self._heap_counter += 1
    heapq.heapify(self._issues_heap)  # Single heapify after batch
```

### 9.6 Batched Statistics Collection

Schema learning collects all statistics in a single `select()` call.

**Implementation**: `src/truthound/schema.py`

```python
# Single select() for null_count, n_unique, etc.
stats_exprs = [
    pl.col(col).null_count().alias(f"{col}_null_count"),
    pl.col(col).n_unique().alias(f"{col}_n_unique"),
    # ...
]
stats = lf.select(stats_exprs).collect()
```

### 9.7 Optimization Summary

| Optimization | Location | Effect |
|-------------|----------|--------|
| Expression Batch Executor | `validators/base.py` | Multiple validators, single collect() |
| Lazy Loading Registry | `validators/_lazy.py` | 200+ validator lazy loading |
| xxhash Cache | `cache.py` | ~10x faster fingerprinting |
| Native Polars Masking | `maskers.py` | Eliminates map_elements |
| Heap-Based Sorting | `report.py` | O(1) severity access |
| Batched Statistics | `schema.py` | Single select() for stats |
| Vectorized Validation | `validators/distribution/range.py` | Vectorized range checks |
| Streaming Mode | `maskers.py` | Streaming for large data |

---

## See Also

- [Architecture Overview](ARCHITECTURE.md) - System design
- [Validators Reference](VALIDATORS.md) - All validators
- [Streaming Validation](ADVANCED.md#streaming) - Streaming details
- [Data Sources](DATASOURCES.md) - Data source optimization
