# Data Profiling Guide

This guide covers automatic data profiling with Truthound's Python API. It includes practical workflows for schema learning, rule generation, and integration with validation pipelines.

---

## Quick Start

```python
import truthound as th

# Profile a file
profile = th.profile("data.csv")
print(f"Rows: {profile.row_count}, Columns: {profile.column_count}")

# Generate validation rules from profile
from truthound.profiler import generate_suite
from truthound.profiler.generators import save_suite

suite = generate_suite(profile)
save_suite(suite, "rules.yaml", format="yaml")
```

---

## Common Workflows

### Workflow 1: Profile and Validate Pipeline

```python
import truthound as th
from truthound.profiler import generate_suite

# Step 1: Profile baseline data
profile = th.profile("baseline.csv")

# Step 2: Generate validation rules
suite = generate_suite(profile, strictness="medium")

# Step 3: Validate new data using generated rules
report = suite.execute(new_data)

# Step 4: Check for issues
if report.issues:
    for issue in report.issues:
        print(f"{issue.column}: {issue.issue_type} ({issue.severity})")
```

### Workflow 2: Schema Evolution Detection

```python
from truthound.profiler import DataProfiler, ProfilerConfig
from truthound.profiler.evolution import SchemaEvolutionDetector

# Profile old and new data
config = ProfilerConfig(sample_size=10000)
profiler = DataProfiler(config)

old_profile = profiler.profile_file("data_v1.csv")
new_profile = profiler.profile_file("data_v2.csv")

# Detect schema changes
detector = SchemaEvolutionDetector()
changes = detector.detect(old_profile, new_profile)

for change in changes:
    print(f"{change.change_type}: {change.description}")
    if change.is_breaking:
        print(f"  WARNING: Breaking change detected!")
```

### Workflow 3: Incremental Profiling with Cache

```python
from truthound.profiler import ProfileCache, IncrementalProfiler

# Setup cache
cache = ProfileCache(cache_dir=".truthound/cache")

# Check if profile is cached
fingerprint = cache.compute_fingerprint("data.csv")
if cache.exists(fingerprint):
    profile = cache.get(fingerprint)
    print("Using cached profile")
else:
    profile = th.profile("data.csv")
    cache.set(fingerprint, profile)
    print("Profile cached for future use")
```

### Workflow 4: Large File Streaming Profile

```python
import polars as pl
from truthound.profiler import StreamingPatternMatcher, IncrementalAggregation

# Setup streaming matcher for pattern detection
matcher = StreamingPatternMatcher(
    aggregation_strategy=IncrementalAggregation(),
    chunk_size=100000,
)

# Process large file in chunks
for chunk in pl.scan_csv("large_file.csv").iter_slices(100000):
    for col in chunk.columns:
        matcher.process_chunk(chunk, col)

# Get aggregated results
results = matcher.finalize()
for col, patterns in results.items():
    print(f"{col}: {patterns.pattern_name} ({patterns.match_ratio:.1%})")
```

---

## Full Documentation

This document describes Truthound's Auto-Profiling system, a streaming profiler with schema versioning and distributed processing capabilities. The system automatically analyzes datasets and generates validation rules without manual configuration.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [CLI Commands](#cli-commands)
5. [Sampling Strategies](#sampling-strategies)
6. [Streaming Pattern Matching](#streaming-pattern-matching)
7. [Rule Generation](#rule-generation)
8. [Caching & Incremental Profiling](#caching--incremental-profiling)
9. [Resilience & Timeout](#resilience--timeout)
10. [Observability](#observability)
11. [Distributed Processing](#distributed-processing)
12. [Profile Comparison & Drift Detection](#profile-comparison--drift-detection)
13. [Quality Scoring](#quality-scoring)
14. [ML-based Type Inference](#ml-based-type-inference)
15. [Automatic Threshold Tuning](#automatic-threshold-tuning)
16. [Profile Visualization](#profile-visualization)
17. [Internationalization (i18n)](#internationalization-i18n)
18. [Incremental Validation](#incremental-validation)
19. [Configuration Reference](#configuration-reference)
20. [API Reference](#api-reference)

---

## Overview

The Auto-Profiling system provides enterprise-grade capabilities for data quality analysis:

- **Automatic Data Profiling**: Column statistics, patterns, and data types with 100M+ rows/sec throughput
- **Rule Generation**: Generate validation rules from profile results with configurable strictness
- **Validation Suite Export**: Export rules to YAML, JSON, Python, or TOML formats
- **Streaming Support**: Process large files in chunks with pattern aggregation strategies
- **Memory Safety**: Sampling strategies and OOM prevention through LazyFrame architecture
- **Process Isolation**: Timeout handling with multiprocessing and circuit breaker patterns
- **Schema Versioning**: Forward/backward compatible profile serialization with automatic migration
- **Caching**: Incremental profiling with fingerprint-based cache invalidation
- **Distributed Processing**: Multi-threaded local processing with framework support for Spark, Dask, and Ray

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Input                                       │
│  CSV / Parquet / JSON / DataFrame / LazyFrame                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Sampling Layer                                      │
│  Random / Stratified / Reservoir / Adaptive / Head / Hash               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Data Profiler                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ColumnProfiler          │  TableProfiler          │  PatternMatcher    │
│  • Basic stats           │  • Row count            │  • Email           │
│  • Null analysis         │  • Column correlations  │  • Phone           │
│  • Distribution          │  • Duplicate detection  │  • UUID            │
│  • Type inference        │  • Schema extraction    │  • Date formats    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Rule Generators                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  SchemaRuleGenerator     │  StatsRuleGenerator     │  PatternRuleGenerator│
│  • Column existence      │  • Range constraints    │  • Regex patterns   │
│  • Type constraints      │  • Null thresholds      │  • Format rules     │
│  • Nullable rules        │  • Uniqueness rules     │  • PII detection    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Suite Export                                         │
│  YAML / JSON / Python / TOML / Checkpoint                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Python API

```python
from truthound.profiler import (
    DataProfiler,
    ProfilerConfig,
    profile_file,
    generate_suite,
)

# Simple profiling
profile = profile_file("data.csv")
print(f"Rows: {profile.row_count}, Columns: {profile.column_count}")

# Generate validation suite
from truthound.profiler.generators import save_suite
suite = generate_suite(profile)
save_suite(suite, "rules.yaml", format="yaml")

# Or use to_yaml() method
with open("rules.yaml", "w") as f:
    f.write(suite.to_yaml())

# Advanced configuration
config = ProfilerConfig(
    sample_size=10000,
    include_patterns=True,
    include_correlations=False,
)
profiler = DataProfiler(config)
profile = profiler.profile_file("large_data.parquet")
```

### CLI

```bash
# Profile a file
truthound auto-profile data.csv -o profile.json

# Generate validation suite from profile
truthound generate-suite profile.json -o rules.yaml

# One-step: profile + generate suite
truthound quick-suite data.csv -o rules.yaml

# List available options
truthound list-formats      # yaml, json, python, toml, checkpoint
truthound list-presets      # default, strict, loose, minimal, comprehensive
truthound list-categories   # schema, stats, pattern, completeness, etc.
```

---

## Core Components

### DataProfiler

The main profiling class that orchestrates column and table analysis.

```python
from truthound.profiler import DataProfiler, ProfilerConfig

config = ProfilerConfig(
    sample_size=50000,            # Max rows to analyze (None = all)
    include_patterns=True,        # Enable pattern detection
    include_correlations=False,   # Compute correlations
    pattern_sample_size=1000,     # Samples for pattern matching
    top_n_values=10,              # Top/bottom values count
)

profiler = DataProfiler(config)

# Profile from various sources
profile = profiler.profile(df.lazy())           # LazyFrame
profile = profiler.profile_file("data.csv")     # File path
profile = profiler.profile_dataframe(df)        # DataFrame
```

### ColumnProfiler

Analyzes individual columns for statistics and patterns.

```python
from truthound.profiler import ColumnProfiler

column_profiler = ColumnProfiler()
col_profile = column_profiler.profile(lf, "email")

print(f"Type: {col_profile.inferred_type}")       # DataType.EMAIL
print(f"Null ratio: {col_profile.null_ratio}")    # 0.02
print(f"Unique ratio: {col_profile.unique_ratio}") # 0.95
print(f"Patterns: {col_profile.patterns}")        # [PatternMatch(...)]
```

### TableProfile

Contains the complete analysis results for a dataset.

```python
@dataclass(frozen=True)
class TableProfile:
    # Table info
    name: str = ""
    row_count: int = 0
    column_count: int = 0

    # Memory estimation
    estimated_memory_bytes: int = 0

    # Column profiles (immutable tuple)
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)

    # Table-level metrics
    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0

    # Correlation matrix (column pairs with high correlation)
    correlations: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    # Metadata
    source: str = ""
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0

    # Methods
    def to_dict(self) -> dict[str, Any]
    def get(self, column_name: str) -> ColumnProfile | None
    @property
    def column_names(self) -> list[str]
```

**Note**: `TableProfile` is immutable (`frozen=True`). Use `to_dict()` and `json.dumps()` for serialization.

---

## CLI Commands

### auto-profile

Profile a data file and output analysis results.

```bash
truthound auto-profile <file> [OPTIONS]

Options:
  -o, --output PATH          Output file path (default: stdout)
  -f, --format [json|yaml]   Output format (default: json)
  --sample-size INTEGER      Maximum rows to sample
  --timeout INTEGER          Timeout in seconds per column
  --no-patterns              Disable pattern detection
  --streaming                Enable streaming mode for large files
  --chunk-size INTEGER       Chunk size for streaming (default: 100000)
```

### generate-suite

Generate validation rules from a profile.

```bash
truthound generate-suite <profile> [OPTIONS]

Options:
  -o, --output PATH          Output file path (required)
  -f, --format FORMAT        Output format: yaml, json, python, toml, checkpoint
  -s, --strictness LEVEL     Strictness: loose, medium, strict
  -p, --preset PRESET        Configuration preset
  --min-confidence LEVEL     Minimum confidence: low, medium, high
  -i, --include CATEGORY     Include only these categories (repeatable)
  -e, --exclude CATEGORY     Exclude these categories (repeatable)
```

### quick-suite

One-step profile and rule generation.

```bash
truthound quick-suite <file> [OPTIONS]

Options:
  -o, --output PATH          Output file path (required)
  -f, --format FORMAT        Output format (default: yaml)
  -s, --strictness LEVEL     Strictness level
  -p, --preset PRESET        Configuration preset
  --sample-size INTEGER      Maximum rows to sample
```

### list-* Commands

```bash
# List available export formats
truthound list-formats
# Output: yaml, json, python, toml, checkpoint

# List available presets
truthound list-presets
# Output: default, strict, loose, minimal, comprehensive, ci_cd, ...

# List rule categories
truthound list-categories
# Output: schema, stats, pattern, completeness, uniqueness, ...
```

---

## Sampling Strategies

The profiler supports multiple sampling strategies for handling large datasets efficiently.

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `NONE` | No sampling, use all data | Small datasets (<100K rows) |
| `HEAD` | Take first N rows | Quick previews |
| `RANDOM` | Random row selection | General use |
| `SYSTEMATIC` | Every Nth row | Ordered data |
| `STRATIFIED` | Maintain distribution | Categorical columns |
| `RESERVOIR` | Reservoir sampling | Streaming data |
| `ADAPTIVE` | Auto-select based on size | Default choice |
| `HASH` | Hash-based deterministic | Reproducibility |

### Usage

```python
from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod

# Configure sampling
config = SamplingConfig(
    strategy=SamplingMethod.ADAPTIVE,
    max_rows=50000,
    confidence_level=0.95,
    margin_of_error=0.01,
)

sampler = Sampler(config)
result = sampler.sample(lf)
sampled_lf = result.data
print(f"Sampled {result.metrics.sample_size} rows")
```

### Memory-Safe Sampling

The sampler automatically prevents OOM by using `.head(limit).collect()`:

```python
# Internal implementation ensures memory safety
def sample(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    # Never calls .collect() on full dataset
    # Uses .head(limit) before collection
    return lf.head(self.config.max_rows)
```

---

## Enterprise-Scale Sampling (100M+ rows)

For datasets exceeding 100 million rows, Truthound provides specialized sampling strategies with O(1) memory footprint.

> **Full Documentation**: See [Enterprise Sampling Guide](enterprise-sampling.md) for complete details on parallel processing, probabilistic data structures, and advanced configurations.

### Quick Start

```python
from truthound.profiler.enterprise_sampling import (
    EnterpriseScaleSampler,
    sample_large_dataset,
)

# Quick sampling with quality preset
result = sample_large_dataset(lf, target_rows=100_000, quality="high")
print(f"Sampled {result.metrics.sample_size:,} rows")

# Auto-select best strategy based on data size
sampler = EnterpriseScaleSampler()
result = sampler.sample(lf)
```

### Scale Categories

| Category | Row Count | Strategy | Memory |
|----------|-----------|----------|--------|
| SMALL | < 1M | No sampling | Full |
| MEDIUM | 1M - 10M | Column-aware | ~500MB |
| LARGE | 10M - 100M | Block (parallel) | ~1GB |
| XLARGE | 100M - 1B | Multi-stage | ~2GB |
| XXLARGE | > 1B | Sketches | O(1) |

### Parallel Block Processing

For maximum throughput with multi-core parallelism:

```python
from truthound.profiler.parallel_sampling import (
    ParallelBlockSampler,
    ParallelSamplingConfig,
    sample_parallel,
)

# Quick parallel sampling
result = sample_parallel(lf, target_rows=100_000, max_workers=4)

# Advanced configuration
config = ParallelSamplingConfig(
    target_rows=100_000,
    max_workers=8,
    enable_work_stealing=True,  # Dynamic load balancing
    backpressure_threshold=0.75,  # Memory-aware scheduling
)
sampler = ParallelBlockSampler(config)
result = sampler.sample(lf)

# Access parallel metrics
print(f"Workers: {result.metrics.workers_used}")
print(f"Speedup: {result.metrics.parallel_speedup:.2f}x")
```

### Probabilistic Data Structures

For 10B+ row datasets, use O(1) memory sketches:

```python
from truthound.profiler.sketches import (
    HyperLogLog,      # Cardinality estimation
    CountMinSketch,   # Frequency estimation
    BloomFilter,      # Membership testing
    create_sketch,
)

# Cardinality estimation (distinct count)
hll = create_sketch("hyperloglog", precision=14)  # ~16KB, ±0.41% error
for chunk in data_stream:
    hll.add_batch(chunk["user_id"])
print(f"Distinct users: ~{hll.estimate():,}")

# Frequency estimation (heavy hitters)
cms = create_sketch("countmin", epsilon=0.001, delta=0.01)
for item in stream:
    cms.add(item)
heavy_hitters = cms.get_heavy_hitters(threshold=0.01)

# Membership testing
bf = create_sketch("bloom", capacity=10_000_000, error_rate=0.001)
bf.add_batch(known_items)
if bf.contains(query_item):
    print("Possibly in set")
```

### Enterprise Sampler Strategies

| Strategy | Best For | Features |
|----------|----------|----------|
| `block` | 10M-100M rows | Parallel, even coverage |
| `multi_stage` | 100M-1B rows | Hierarchical reduction |
| `column_aware` | Mixed types | Type-weighted sampling |
| `progressive` | Exploratory | Early stopping |
| `parallel_block` | High throughput | Work stealing |

```python
from truthound.profiler.enterprise_sampling import EnterpriseScaleSampler

sampler = EnterpriseScaleSampler(config)

# Auto-select best strategy
result = sampler.sample(lf)

# Or specify explicitly
result = sampler.sample(lf, strategy="block")
result = sampler.sample(lf, strategy="multi_stage")
```

### Validators with Large Data Support

```python
from truthound.validators.base import Validator, EnterpriseScaleSamplingMixin

class MyLargeDataValidator(Validator, EnterpriseScaleSamplingMixin):
    sampling_threshold = 10_000_000
    sampling_target_rows = 100_000

    def validate(self, lf):
        sampled_lf, metrics = self._sample_for_validation(lf)
        issues = self._do_validation(sampled_lf)
        if metrics.is_sampled:
            issues = self._extrapolate_issues(issues, metrics)
        return issues
```

---

## Streaming Pattern Matching

For large files that don't fit in memory, use streaming mode with chunk-aware pattern aggregation.

### Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `INCREMENTAL` | Running totals across chunks |
| `WEIGHTED` | Size-weighted averages |
| `SLIDING_WINDOW` | Recent chunks only |
| `EXPONENTIAL` | Exponential moving average |
| `CONSENSUS` | Agreement across chunks |
| `ADAPTIVE` | Auto-select based on variance |

### Usage

```python
from truthound.profiler import StreamingPatternMatcher, IncrementalAggregation

matcher = StreamingPatternMatcher(
    aggregation_strategy=IncrementalAggregation(),
    chunk_size=100000,
)

# Process file in chunks
for chunk in pl.scan_csv("large.csv").iter_slices(100000):
    matcher.process_chunk(chunk, "email")

# Get aggregated results
results = matcher.finalize()
print(f"Email pattern match: {results['email'].match_ratio:.2%}")
```

### CLI Streaming Mode

```bash
truthound auto-profile large_file.csv --streaming --chunk-size 100000
```

---

## Rule Generation

### Rule Categories

| Category | Description | Example Rules |
|----------|-------------|---------------|
| `schema` | Column structure | Column exists, type check |
| `stats` | Statistical constraints | Range, mean bounds |
| `pattern` | Format validation | Email, phone, UUID |
| `completeness` | Null handling | Max null ratio |
| `uniqueness` | Duplicate detection | Primary key, unique |
| `distribution` | Value distribution | Allowed values, cardinality |

### Strictness Levels

| Level | Description |
|-------|-------------|
| `loose` | Permissive thresholds, fewer rules |
| `medium` | Balanced defaults |
| `strict` | Tight thresholds, comprehensive rules |

### Presets

| Preset | Use Case |
|--------|----------|
| `default` | General purpose |
| `strict` | Production data |
| `loose` | Development/testing |
| `minimal` | Essential rules only |
| `comprehensive` | All available rules |
| `ci_cd` | Optimized for CI/CD pipelines |
| `schema_only` | Structure validation only |
| `format_only` | Format/pattern rules only |

### Export Formats

```python
from truthound.profiler import generate_suite
from truthound.profiler.generators import save_suite

suite = generate_suite(profile)

# YAML (human-readable)
save_suite(suite, "rules.yaml", format="yaml")

# JSON (machine-readable)
save_suite(suite, "rules.json", format="json")

# Python (executable)
save_suite(suite, "rules.py", format="python")

# Or use built-in methods
with open("rules.yaml", "w") as f:
    f.write(suite.to_yaml())

import json
with open("rules.json", "w") as f:
    json.dump(suite.to_dict(), f, indent=2)
```

---

## Caching & Incremental Profiling

### Fingerprint-Based Caching

The profiler caches results based on file fingerprints:

```python
from truthound.profiler import ProfileCache

cache = ProfileCache(cache_dir=".truthound/cache")

# Check if profile is cached
fingerprint = cache.compute_fingerprint("data.csv")
if cache.exists(fingerprint):
    profile = cache.get(fingerprint)
else:
    profile = profiler.profile_file("data.csv")
    cache.set(fingerprint, profile)
```

### Incremental Profiling

Only re-profile columns that have changed:

```python
from truthound.profiler import IncrementalProfiler

inc_profiler = IncrementalProfiler()

# Initial profile
profile_v1 = inc_profiler.profile("data_v1.csv")

# Incremental update (only changed columns)
profile_v2 = inc_profiler.update("data_v2.csv", previous=profile_v1)

print(f"Columns re-profiled: {profile_v2.columns_updated}")
print(f"Columns reused: {profile_v2.columns_cached}")
```

### Cache Backends

| Backend | Description |
|---------|-------------|
| `memory` | In-memory LRU cache |
| `file` | Disk-based JSON cache |
| `redis` | Redis backend (optional) |

```python
from truthound.profiler import CacheBackend, RedisCacheBackend

# Redis cache with fallback
cache = RedisCacheBackend(
    host="redis.example.com",
    fallback=FileCacheBackend(".truthound/cache"),
)
```

---

## Resilience & Timeout

### Process Isolation

Polars operations can't be interrupted by Python signals. The profiler uses process isolation:

```python
from truthound.profiler import ProcessTimeoutExecutor

executor = ProcessTimeoutExecutor(
    timeout_seconds=60,
    backend="process",  # Use multiprocessing
)

# Safely profile with timeout
try:
    result = executor.execute(profiler.profile, lf)
except TimeoutError:
    print("Profiling timed out")
```

### Circuit Breaker

Prevent cascade failures with circuit breaker pattern:

```python
from truthound.profiler import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30,      # Try again after 30s
    success_threshold=2,      # Close after 2 successes
))

with breaker:
    result = risky_operation()
```

### Resilient Cache

Automatic fallback between cache backends:

```python
from truthound.profiler import ResilientCacheBackend, FallbackChain

cache = ResilientCacheBackend(
    primary=RedisCacheBackend(host="redis"),
    fallback=FileCacheBackend(".cache"),
    circuit_breaker=CircuitBreakerConfig(failure_threshold=3),
)
```

---

## Observability

### OpenTelemetry Integration

```python
from truthound.profiler import ProfilerTelemetry, traced

# Initialize telemetry
telemetry = ProfilerTelemetry(service_name="truthound-profiler")

# Automatic tracing with decorator
@traced("profile_column")
def profile_column(lf: pl.LazyFrame, column: str):
    # Profiling logic
    pass

# Manual span creation
with telemetry.span("custom_operation") as span:
    span.set_attribute("column_count", 10)
    # Operation
```

### Metrics Collection

```python
from truthound.profiler import MetricsCollector

collector = MetricsCollector()

# Record profile duration
with collector.profile_duration.time(operation="column"):
    profile_column(lf, "email")

# Increment counters
collector.profiles_total.inc(status="success")
collector.columns_profiled.inc(data_type="string")
```

### Progress Callbacks

```python
from truthound.profiler import CallbackRegistry, ConsoleAdapter

# Console progress output
callback = ConsoleAdapter(show_progress=True)

profiler = DataProfiler(config, progress_callback=callback)
profile = profiler.profile_file("data.csv")

# Custom callback
def my_callback(event):
    print(f"Progress: {event.progress:.0%} - {event.message}")

profiler = DataProfiler(config, progress_callback=my_callback)
```

---

## Distributed Processing

The distributed processing module enables parallel profiling across multiple cores or nodes, achieving linear scalability for large datasets.

### Available Backends

| Backend | Description | Execution Strategy | Status |
|---------|-------------|-------------------|--------|
| `local` | Multi-threaded local | ThreadPoolExecutor with adaptive sizing | ✅ Ready |
| `process` | Multi-process local | ProcessPoolExecutor with isolation | ✅ Ready |
| `spark` | Apache Spark | SparkContext-based distribution | ⚠️ Framework |
| `dask` | Dask distributed | Delayed computation graph | ⚠️ Framework |
| `ray` | Ray framework | Actor-based parallelism | ⚠️ Framework |

### Usage

```python
from truthound.profiler import DistributedProfiler, BackendType

# Auto-detect available backend
profiler = DistributedProfiler.create(backend="auto")

# Explicit Spark backend
profiler = DistributedProfiler.create(
    backend="spark",
    config={"spark.executor.memory": "4g"}
)

# Profile large dataset
profile = profiler.profile("hdfs://data/large.parquet")
```

### Partition Strategies

| Strategy | Description | Optimal Use Case |
|----------|-------------|------------------|
| `row_based` | Split by row ranges | Large row counts, uniform columns |
| `column_based` | Profile columns in parallel | Many columns, limited rows |
| `hybrid` | Combine both strategies | Balanced datasets |
| `hash` | Hash-based partitioning | Reproducible results |

### Execution Backend Selection

The `AdaptiveStrategy` automatically selects the optimal execution backend based on operation characteristics:

```python
from truthound.profiler import AdaptiveStrategy, ExecutionBackend

strategy = AdaptiveStrategy()

# Automatic selection based on data size and operation type
backend = strategy.select(
    estimated_rows=10_000_000,
    operation_type="profile_column",
    cpu_bound=True,
)
# Returns: ExecutionBackend.THREAD for I/O-bound
# Returns: ExecutionBackend.PROCESS for CPU-bound with large data
```

---

## Profile Comparison & Drift Detection

Compare profiles over time to detect data drift and schema changes.

### Drift Types

| Drift Type | Description |
|------------|-------------|
| `COMPLETENESS` | Changes in null ratios |
| `UNIQUENESS` | Changes in unique value ratios |
| `DISTRIBUTION` | Changes in value distribution |
| `RANGE` | Changes in min/max values |
| `CARDINALITY` | Changes in distinct value counts |

### Drift Severity

| Severity | Description |
|----------|-------------|
| `LOW` | Minor changes, informational |
| `MEDIUM` | Notable changes, review recommended |
| `HIGH` | Significant changes, investigation required |
| `CRITICAL` | Breaking changes, immediate action needed |

### Usage

```python
from truthound.profiler import (
    ProfileComparator,
    compare_profiles,
    detect_drift,
    DriftThresholds,
)

# Compare two profiles
comparator = ProfileComparator()
comparison = comparator.compare(profile_old, profile_new)

print(f"Overall drift score: {comparison.drift_score:.2%}")
print(f"Schema changes: {len(comparison.schema_changes)}")

for col_comparison in comparison.column_comparisons:
    if col_comparison.has_drift:
        print(f"Column {col_comparison.column}: {col_comparison.drift_type}")
        for drift in col_comparison.drifts:
            print(f"  - {drift.drift_type}: {drift.severity}")

# Quick drift detection
drifts = detect_drift(profile_old, profile_new)
for drift in drifts:
    print(f"{drift.column}: {drift.drift_type} ({drift.severity})")
```

### Custom Thresholds

```python
from truthound.profiler import DriftThresholds

thresholds = DriftThresholds(
    completeness_change=0.05,    # 5% change in null ratio
    uniqueness_change=0.1,       # 10% change in uniqueness
    distribution_divergence=0.2,  # Distribution KL-divergence
    range_change=0.15,           # 15% change in range
    cardinality_change=0.2,      # 20% change in cardinality
)

comparison = compare_profiles(profile_old, profile_new, thresholds=thresholds)
```

### Individual Drift Detectors

```python
from truthound.profiler import (
    CompletenessDriftDetector,
    UniquenessDriftDetector,
    DistributionDriftDetector,
    RangeDriftDetector,
    CardinalityDriftDetector,
)

# Detect specific drift types
completeness_detector = CompletenessDriftDetector(threshold=0.05)
drift = completeness_detector.detect(col_profile_old, col_profile_new)

if drift.detected:
    print(f"Null ratio changed: {drift.old_value:.2%} -> {drift.new_value:.2%}")
```

---

## Quality Scoring

Evaluate and score generated validation rules for quality.

### Quality Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| `EXCELLENT` | 0.9 - 1.0 | High precision and recall |
| `GOOD` | 0.7 - 0.9 | Balanced quality |
| `FAIR` | 0.5 - 0.7 | Acceptable with caveats |
| `POOR` | 0.0 - 0.5 | Needs improvement |

### Usage

```python
from truthound.profiler import (
    RuleQualityScorer,
    ScoringConfig,
    estimate_quality,
    score_rule,
    compare_rules,
)

# Score a validation rule
scorer = RuleQualityScorer()
score = scorer.score(rule, sample_data)

print(f"Quality level: {score.quality_level}")
print(f"Precision: {score.metrics.precision:.2%}")
print(f"Recall: {score.metrics.recall:.2%}")
print(f"F1 Score: {score.metrics.f1_score:.2%}")

# Quick quality estimation
quality = estimate_quality(rule, lf)
print(f"Estimated quality: {quality.level}")
```

### Quality Estimators

```python
from truthound.profiler import (
    SamplingQualityEstimator,
    HeuristicQualityEstimator,
    CrossValidationEstimator,
)

# Sampling-based estimation (fast)
estimator = SamplingQualityEstimator(sample_size=1000)
result = estimator.estimate(rule, lf)

# Cross-validation (accurate)
estimator = CrossValidationEstimator(n_folds=5)
result = estimator.estimate(rule, lf)

# Heuristic-based (fastest)
estimator = HeuristicQualityEstimator()
result = estimator.estimate(rule, lf)
```

### Trend Analysis

```python
from truthound.profiler import QualityTrendAnalyzer

analyzer = QualityTrendAnalyzer()

# Track quality over time
for score in historical_scores:
    analyzer.add_point(score)

trend = analyzer.analyze()
print(f"Trend direction: {trend.direction}")  # IMPROVING, STABLE, DECLINING
print(f"Forecast: {trend.forecast_quality}")
```

---

## ML-based Type Inference

Use machine learning models for improved semantic type inference.

### Inference Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `RuleBasedModel` | Pattern matching rules | Fast, deterministic |
| `NaiveBayesModel` | Probabilistic classifier | Good for mixed types |
| `EnsembleModel` | Combines multiple models | Best accuracy |

### Usage

```python
from truthound.profiler import (
    MLTypeInferrer,
    infer_column_type_ml,
    infer_table_types_ml,
    MLInferenceConfig,
)

# Configure ML inferrer
config = MLInferenceConfig(
    model="ensemble",
    confidence_threshold=0.7,
    use_name_features=True,
    use_value_features=True,
    use_statistical_features=True,
)

inferrer = MLTypeInferrer(config)

# Infer single column type
result = inferrer.infer(lf, "email_address")
print(f"Type: {result.inferred_type}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Alternatives: {result.alternatives}")

# Infer all column types
results = infer_table_types_ml(lf)
for col, result in results.items():
    print(f"{col}: {result.inferred_type} ({result.confidence:.0%})")
```

### Feature Extractors

```python
from truthound.profiler import (
    NameFeatureExtractor,
    ValueFeatureExtractor,
    StatisticalFeatureExtractor,
    ContextFeatureExtractor,
)

# Extract features for custom models
name_extractor = NameFeatureExtractor()
features = name_extractor.extract("customer_email")
# Features: ['has_email_keyword', 'has_address_keyword', ...]

value_extractor = ValueFeatureExtractor()
features = value_extractor.extract(column_values)
# Features: ['has_at_symbol', 'avg_length', 'digit_ratio', ...]
```

### Custom Models

```python
from truthound.profiler import InferenceModel, InferenceResult

class MyCustomModel(InferenceModel):
    def infer(self, features: FeatureVector) -> InferenceResult:
        # Custom inference logic
        return InferenceResult(
            inferred_type=DataType.EMAIL,
            confidence=0.95,
        )

# Register custom model
from truthound.profiler import InferenceModelRegistry
InferenceModelRegistry.register("my_model", MyCustomModel)
```

---

## Automatic Threshold Tuning

Automatically tune validation thresholds based on data characteristics.

### Tuning Strategies

| Strategy | Description |
|----------|-------------|
| `CONSERVATIVE` | Strict thresholds, fewer false positives |
| `BALANCED` | Balance between precision and recall |
| `PERMISSIVE` | Relaxed thresholds, fewer false negatives |
| `ADAPTIVE` | Learns optimal thresholds from data |
| `STATISTICAL` | Uses statistical confidence intervals |
| `DOMAIN_AWARE` | Incorporates domain knowledge |

### Usage

```python
from truthound.profiler import (
    ThresholdTuner,
    tune_thresholds,
    TuningStrategy,
    TuningConfig,
)

# Quick threshold tuning
thresholds = tune_thresholds(profile, strategy="adaptive")

print(f"Null threshold: {thresholds.null_threshold:.2%}")
print(f"Uniqueness threshold: {thresholds.uniqueness_threshold:.2%}")
print(f"Range tolerance: {thresholds.range_tolerance:.2%}")

# Advanced configuration
config = TuningConfig(
    strategy=TuningStrategy.STATISTICAL,
    confidence_level=0.95,
    sample_size=10000,
)

tuner = ThresholdTuner(config)
thresholds = tuner.tune(profile)

# Per-column thresholds
for col, col_thresholds in thresholds.column_thresholds.items():
    print(f"{col}:")
    print(f"  null_threshold: {col_thresholds.null_threshold}")
    print(f"  min_value: {col_thresholds.min_value}")
    print(f"  max_value: {col_thresholds.max_value}")
```

### A/B Testing Thresholds

```python
from truthound.profiler import ThresholdTester, ThresholdTestResult

tester = ThresholdTester()

# Test different threshold configurations
result = tester.test(
    profile=profile,
    test_data=lf,
    threshold_configs=[config_a, config_b, config_c],
)

print(f"Best config: {result.best_config}")
print(f"Precision: {result.best_precision:.2%}")
print(f"Recall: {result.best_recall:.2%}")
```

---

## Profile Visualization

Generate HTML reports from profile results.

### Chart Types

| Type | Description |
|------|-------------|
| `BAR` | Bar charts for categorical data |
| `HISTOGRAM` | Distribution histograms |
| `PIE` | Pie charts for proportions |
| `LINE` | Trend lines |
| `HEATMAP` | Correlation heatmaps |

### Themes

| Theme | Description |
|-------|-------------|
| `LIGHT` | Light background theme |
| `DARK` | Dark background theme |
| `CORPORATE` | Professional business theme |
| `MINIMAL` | Clean minimal design |
| `COLORFUL` | Vibrant color scheme |

### Usage

```python
from truthound.profiler import (
    HTMLReportGenerator,
    ReportConfig,
    ReportTheme,
    generate_report,
)

# Quick report generation
html = generate_report(profile)
with open("report.html", "w") as f:
    f.write(html)

# Custom configuration
config = ReportConfig(
    theme=ReportTheme.DARK,
    include_charts=True,
    include_samples=True,
    include_recommendations=True,
    max_sample_rows=100,
)

generator = HTMLReportGenerator(config)
html = generator.generate(profile)

# Export to multiple formats
from truthound.profiler import ReportExporter

exporter = ReportExporter()
exporter.export(profile, "report.html", format="html")
exporter.export(profile, "report.pdf", format="pdf")
```

### Section Renderers

```python
from truthound.profiler import (
    OverviewSectionRenderer,
    DataQualitySectionRenderer,
    ColumnDetailsSectionRenderer,
    PatternsSectionRenderer,
    RecommendationsSectionRenderer,
    CustomSectionRenderer,
)

# Add custom section
class MySectionRenderer(CustomSectionRenderer):
    section_type = "my_section"

    def render(self, data: ProfileData) -> str:
        return f"<div class='my-section'>Custom content</div>"

from truthound.profiler import section_registry
section_registry.register(MySectionRenderer())
```

### Compare Profiles Report

```python
from truthound.profiler import compare_profile_reports

# Generate comparison report
html = compare_profile_reports(profile_old, profile_new)
with open("comparison_report.html", "w") as f:
    f.write(html)
```

---

## Internationalization (i18n)

Multi-language support for profiler messages and reports.

### Supported Locales

| Locale | Language |
|--------|----------|
| `en` | English (default) |
| `ko` | Korean |
| `ja` | Japanese |
| `zh` | Chinese |
| `de` | German |
| `fr` | French |
| `es` | Spanish |

### Usage

```python
from truthound.profiler import (
    set_locale,
    get_locale,
    get_message,
    translate as t,
    locale_context,
)

# Set global locale
set_locale("ko")

# Get translated message
msg = get_message("profiling_started")
print(msg)  # "프로파일링이 시작되었습니다"

# Short alias
msg = t("column_null_ratio", column="email", ratio=0.05)

# Context manager for temporary locale
with locale_context("ja"):
    msg = t("profiling_complete")
    print(msg)  # Japanese message
```

### Custom Messages

```python
from truthound.profiler import register_messages, MessageCatalog

# Register custom messages
custom_messages = {
    "my_custom_key": "Custom message with {placeholder}",
}
register_messages("en", custom_messages)

# Load from file
from truthound.profiler import load_messages_from_file
load_messages_from_file("my_messages.yaml", locale="ko")
```

### I18n Exceptions

```python
from truthound.profiler import (
    I18nError,
    I18nAnalysisError,
    I18nPatternError,
    I18nTypeError,
    I18nIOError,
    I18nTimeoutError,
    I18nValidationError,
)

try:
    profile = profiler.profile_file("data.csv")
except I18nAnalysisError as e:
    # Error message is automatically localized
    print(e.localized_message)
```

---

## Incremental Validation

Validate incremental profiling results for correctness.

### Validators

| Validator | Purpose |
|-----------|---------|
| `ChangeDetectionAccuracyValidator` | Validates change detection accuracy |
| `SchemaChangeValidator` | Validates schema change detection |
| `StalenessValidator` | Validates profile freshness |
| `FingerprintConsistencyValidator` | Validates fingerprint consistency |
| `FingerprintSensitivityValidator` | Validates fingerprint sensitivity |
| `ProfileMergeValidator` | Validates profile merging |
| `DataIntegrityValidator` | Validates data integrity |
| `PerformanceValidator` | Validates performance metrics |

### Usage

```python
from truthound.profiler import (
    IncrementalValidator,
    validate_incremental,
    validate_merge,
    validate_fingerprints,
    ValidationConfig,
)

# Validate incremental profiling
result = validate_incremental(
    old_profile=profile_v1,
    new_profile=profile_v2,
    new_data=lf_new,
)

if result.passed:
    print("Incremental profiling validated successfully")
else:
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.message}")

# Validate merge operation
result = validate_merge(
    profile_a=part1_profile,
    profile_b=part2_profile,
    merged_profile=merged,
)

# Validate fingerprints
result = validate_fingerprints(
    fingerprints=[fp1, fp2, fp3],
    expected_columns=["email", "phone", "name"],
)
```

### Custom Validators

```python
from truthound.profiler import BaseValidator, ValidatorProtocol

class MyCustomValidator(BaseValidator):
    name = "my_validator"
    category = ValidationCategory.DATA_INTEGRITY

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        issues = []
        # Custom validation logic
        return issues

# Register validator
from truthound.profiler import validator_registry, register_validator
register_validator(MyCustomValidator())
```

### Validation Runner

```python
from truthound.profiler import ValidationRunner, ValidationConfig

config = ValidationConfig(
    validators=["change_detection", "schema_change", "staleness"],
    fail_fast=False,
    severity_threshold="medium",
)

runner = ValidationRunner(config)
result = runner.run(context)

print(f"Total issues: {result.total_issues}")
print(f"Critical: {result.critical_count}")
print(f"High: {result.high_count}")
print(f"Passed: {result.passed}")
```

---

## Configuration Reference

### ProfilerConfig

```python
@dataclass
class ProfilerConfig:
    """Configuration for profiling operations."""

    # Sampling
    sample_size: int | None = None    # None = use all data
    random_seed: int = 42

    # Analysis options
    include_patterns: bool = True     # Detect patterns in string columns
    include_correlations: bool = False  # Compute column correlations
    include_distributions: bool = True  # Include distribution stats

    # Performance tuning
    top_n_values: int = 10            # Top/bottom values to include
    pattern_sample_size: int = 1000   # Sample size for pattern matching
    correlation_threshold: float = 0.7

    # Pattern detection
    min_pattern_match_ratio: float = 0.8

    # Parallel processing
    n_jobs: int = 1
```

**Note**: Timeout and caching are handled separately through `ProcessTimeoutExecutor` and `ProfileCache` classes.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_CACHE_DIR` | Cache directory | `.truthound/cache` |
| `TRUTHOUND_SAMPLE_SIZE` | Default sample size | `50000` |
| `TRUTHOUND_TIMEOUT` | Default timeout (seconds) | `120` |
| `TRUTHOUND_LOG_LEVEL` | Logging level | `INFO` |

---

## API Reference

### Main Functions

```python
from truthound.profiler import (
    profile_file,
    generate_suite,
    DataProfiler,
    ProfilerConfig,
)
from truthound.profiler.generators import save_suite

# Profile a file
profile = profile_file(
    path: str,
    config: ProfilerConfig | None = None,
) -> TableProfile

# Profile a DataFrame/LazyFrame using DataProfiler
profiler = DataProfiler(config)
profile = profiler.profile(lf: pl.LazyFrame) -> TableProfile
profile = profiler.profile_dataframe(df: pl.DataFrame) -> TableProfile

# Generate validation suite
suite = generate_suite(
    profile: TableProfile | ProfileReport | dict,
    strictness: Strictness = Strictness.MEDIUM,
    preset: str = "default",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> ValidationSuite

# Save suite to file
save_suite(
    suite: ValidationSuite,
    path: str | Path,
    format: str = "json",  # "json", "yaml", or "python"
) -> None

# Profile serialization
import json
with open("profile.json", "w") as f:
    json.dump(profile.to_dict(), f, indent=2)
```

**Note**: There is no `quick_suite()` function. Use `profile_file()` + `generate_suite()` + `save_suite()` instead.

### Data Types

```python
class DataType(str, Enum):
    """Inferred logical data types for profiling."""

    # Basic types
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"

    # Semantic types (detected from patterns)
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    JSON = "json"

    # Identifiers
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"

    # Numeric subtypes
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

    # Korean specific
    KOREAN_RRN = "korean_rrn"
    KOREAN_PHONE = "korean_phone"
    KOREAN_BUSINESS_NUMBER = "korean_business_number"

    # Unknown
    UNKNOWN = "unknown"
```

### Strictness Levels

```python
class Strictness(Enum):
    LOOSE = "loose"      # Permissive thresholds
    MEDIUM = "medium"    # Balanced defaults
    STRICT = "strict"    # Tight constraints
```

---

## Examples

### Basic Profiling Pipeline

```python
from truthound.profiler import (
    DataProfiler,
    ProfilerConfig,
    generate_suite,
)

# Configure profiler
config = ProfilerConfig(
    sample_size=100000,
    include_patterns=True,
)

# Profile data
profiler = DataProfiler(config)
profile = profiler.profile_file("sales_data.csv")

# Print summary
print(f"Dataset: {profile.name}")
print(f"Rows: {profile.row_count:,}")
print(f"Columns: {profile.column_count}")

for col in profile.columns:
    print(f"  {col.name}: {col.inferred_type.value}")
    print(f"    Null ratio: {col.null_ratio:.2%}")
    if col.patterns:
        print(f"    Pattern: {col.patterns[0].pattern_name}")

# Generate and save rules
from truthound.profiler.generators import save_suite
suite = generate_suite(profile)
save_suite(suite, "sales_rules.yaml", format="yaml")
```

### Streaming Large Files

```python
from truthound.profiler import (
    StreamingPatternMatcher,
    AdaptiveAggregation,
)
import polars as pl

# Setup streaming matcher
matcher = StreamingPatternMatcher(
    aggregation_strategy=AdaptiveAggregation(),
)

# Process 10GB file in chunks
for chunk in pl.scan_csv("huge_file.csv").iter_slices(100000):
    for col in chunk.columns:
        matcher.process_chunk(chunk, col)

# Get final aggregated patterns
results = matcher.finalize()
```

### CI/CD Integration

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on: [push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Generate validation rules
        run: |
          truthound quick-suite data/sample.csv \
            -o rules.yaml \
            --preset ci_cd \
            --strictness strict

      - name: Validate data
        run: |
          truthound check data/production.csv \
            --schema rules.yaml \
            --strict
```

---

## Schema Versioning

The profiler implements semantic versioning for profile serialization, ensuring forward and backward compatibility across versions.

### Version Format

```
major.minor.patch
```

- **major**: Breaking changes requiring migration
- **minor**: New features, backward compatible
- **patch**: Bug fixes, fully compatible

### Automatic Migration

```python
from truthound.profiler import ProfileSerializer, SchemaVersion

serializer = ProfileSerializer()

# Load profile from any compatible version
profile = serializer.deserialize(old_profile_data)
# Automatically migrates from source version to current

# Save with current schema version
data = serializer.serialize(profile)
# Includes schema_version field for future compatibility
```

### Validation

```python
from truthound.profiler import SchemaValidator, ValidationResult

validator = SchemaValidator()
result = validator.validate(profile_data, fix_issues=True)

if result.result == ValidationResult.VALID:
    print("Profile data is valid")
elif result.result == ValidationResult.RECOVERABLE:
    print(f"Fixed issues: {result.warnings}")
    data = result.fixed_data
else:
    print(f"Invalid: {result.errors}")
```

---

## See Also

- [Schema Evolution](schema-evolution.md) - Schema versioning, history, and monitoring
- [Drift Detection](drift-detection.md) - Detect data distribution changes
- [Quality Scoring](quality-scoring.md) - Evaluate data quality
- [Validators Reference](VALIDATORS.md)
- [Statistical Methods](STATISTICAL_METHODS.md)
- [Checkpoint & CI/CD](CHECKPOINT.md)
- [Storage Backends](STORES.md)
- [Examples](EXAMPLES.md)
