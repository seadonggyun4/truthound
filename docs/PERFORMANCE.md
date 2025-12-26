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

### Test Environment

| Specification | Value |
|---------------|-------|
| CPU | Apple M1 Pro (10 cores) |
| RAM | 32GB |
| Storage | NVMe SSD |
| Python | 3.11 |
| Polars | 0.20.x |

### Single Validator Performance

#### Null Check Validator

| Rows | Columns | Time (ms) | Memory (MB) | Throughput (M rows/s) |
|------|---------|-----------|-------------|----------------------|
| 100K | 10 | 5 | 8 | 20.0 |
| 1M | 10 | 45 | 80 | 22.2 |
| 10M | 10 | 420 | 800 | 23.8 |
| 100M | 10 | 4,200 | 8,000 | 23.8 |

#### Range Validator

| Rows | Columns | Time (ms) | Memory (MB) | Throughput (M rows/s) |
|------|---------|-----------|-------------|----------------------|
| 100K | 10 | 8 | 8 | 12.5 |
| 1M | 10 | 65 | 80 | 15.4 |
| 10M | 10 | 600 | 800 | 16.7 |
| 100M | 10 | 5,800 | 8,000 | 17.2 |

#### Foreign Key Validator (with sampling)

| Child Rows | Parent Rows | Sample Size | Time (ms) | Memory (MB) |
|------------|-------------|-------------|-----------|-------------|
| 1M | 100K | Full | 250 | 120 |
| 10M | 100K | Full | 2,400 | 1,200 |
| 10M | 100K | 100K | 280 | 150 |
| 100M | 1M | 100K | 320 | 180 |

### Multi-Validator Performance

Running multiple validators in a single pass:

| Validators | Rows | Time (ms) | vs Sequential | Memory (MB) |
|------------|------|-----------|---------------|-------------|
| 5 | 1M | 180 | 1.0x (baseline) | 100 |
| 5 | 1M | 150 | 0.83x (optimized) | 100 |
| 10 | 10M | 1,800 | 1.0x (baseline) | 900 |
| 10 | 10M | 1,200 | 0.67x (optimized) | 900 |

### Streaming Validation Performance

| File Size | Chunk Size | Time (s) | Peak Memory (MB) | Throughput |
|-----------|------------|----------|------------------|------------|
| 1GB | 100K rows | 12 | 250 | 83 MB/s |
| 10GB | 100K rows | 115 | 280 | 87 MB/s |
| 100GB | 100K rows | 1,150 | 300 | 87 MB/s |
| 1TB | 100K rows | 11,500 | 350 | 87 MB/s |

### Drift Detection Performance

| Baseline Rows | Current Rows | Method | Time (ms) |
|---------------|--------------|--------|-----------|
| 100K | 100K | KS Test | 85 |
| 1M | 1M | KS Test | 750 |
| 100K | 100K | PSI | 45 |
| 1M | 1M | PSI | 380 |
| 100K | 100K | Chi-Square | 35 |
| 1M | 1M | Chi-Square | 280 |

### ML Anomaly Detection Performance

| Rows | Features | Algorithm | Sample Size | Time (ms) | Memory (MB) |
|------|----------|-----------|-------------|-----------|-------------|
| 10K | 10 | IsolationForest | Full | 120 | 15 |
| 100K | 10 | IsolationForest | 10K | 140 | 18 |
| 1M | 10 | IsolationForest | 10K | 180 | 25 |
| 10K | 10 | LocalOutlierFactor | Full | 450 | 20 |
| 100K | 10 | LocalOutlierFactor | 5K | 220 | 15 |

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

## See Also

- [Architecture Overview](ARCHITECTURE.md) - System design
- [Validators Reference](VALIDATORS.md) - All validators
- [Streaming Validation](ADVANCED.md#streaming) - Streaming details
- [Data Sources](DATASOURCES.md) - Data source optimization
