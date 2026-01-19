# Auto-Profiling & Rule Generation (Phase 7)

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
12. [Configuration Reference](#configuration-reference)
13. [API Reference](#api-reference)

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
suite = generate_suite(profile)
suite.save("rules.yaml")

# Advanced configuration
config = ProfilerConfig(
    sample_size=10000,
    timeout_seconds=60,
    enable_pattern_detection=True,
    strictness="medium",
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
    sample_size=50000,           # Max rows to analyze
    timeout_seconds=120,         # Timeout per column
    enable_caching=True,         # Use fingerprint cache
    enable_pattern_detection=True,
    pattern_sample_size=1000,    # Samples for pattern matching
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
@dataclass
class TableProfile:
    name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    profiled_at: datetime
    duration_ms: float

    # Optional analysis
    duplicate_rows: int | None
    correlations: dict[tuple[str, str], float] | None

    def to_dict(self) -> dict
    def to_json(self) -> str
    def save(self, path: str) -> None
```

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
from truthound.profiler import Sampler, SamplingConfig, SamplingStrategy

# Configure sampling
config = SamplingConfig(
    strategy=SamplingStrategy.ADAPTIVE,
    max_rows=50000,
    confidence_level=0.95,
    margin_of_error=0.01,
)

sampler = Sampler(config)
sampled_lf = sampler.sample(lf)
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

### Scale Categories

| Category | Row Count | Recommended Strategy |
|----------|-----------|---------------------|
| SMALL | < 1M | No sampling needed |
| MEDIUM | 1M - 10M | Systematic or adaptive |
| LARGE | 10M - 100M | Block sampling |
| XLARGE | 100M - 1B | Multi-stage sampling |
| XXLARGE | > 1B | Probabilistic sketches |

### Enterprise Sampling Strategies

#### Block Sampling

Processes data in fixed-size blocks with O(1) memory:

```python
from truthound.profiler.enterprise_sampling import (
    BlockSamplingStrategy,
    EnterpriseScaleConfig,
)

config = EnterpriseScaleConfig(
    target_rows=100_000,
    memory_budget_mb=512,
)
strategy = BlockSamplingStrategy(config)
result = strategy.sample(lf, base_config)
```

#### Multi-Stage Sampling

Hierarchical sampling for billion-row datasets:

```python
from truthound.profiler.enterprise_sampling import MultiStageSamplingStrategy

strategy = MultiStageSamplingStrategy(config, num_stages=3)
result = strategy.sample(lf, base_config)
```

#### Column-Aware Sampling

Weights samples based on column types and cardinality:

```python
from truthound.profiler.enterprise_sampling import ColumnAwareSamplingStrategy

strategy = ColumnAwareSamplingStrategy(config)
result = strategy.sample(lf, base_config)
```

#### Progressive Sampling

Iteratively refines sample until convergence:

```python
from truthound.profiler.enterprise_sampling import ProgressiveSamplingStrategy

strategy = ProgressiveSamplingStrategy(
    config,
    convergence_threshold=0.01,
    max_stages=5,
)
result = strategy.sample(lf, base_config)
```

### Enterprise Sampler Interface

```python
from truthound.profiler.enterprise_sampling import (
    EnterpriseScaleSampler,
    EnterpriseScaleConfig,
    MemoryBudgetConfig,
    SamplingQuality,
)

# Configure for enterprise scale
config = EnterpriseScaleConfig(
    target_rows=100_000,
    memory_budget=MemoryBudgetConfig(max_memory_mb=1024),
    time_budget_seconds=60.0,
    quality=SamplingQuality.STANDARD,
    confidence_level=0.95,
)

# Create sampler
sampler = EnterpriseScaleSampler(config)

# Auto-select best strategy
result = sampler.sample(lf)

# Or specify strategy explicitly
result = sampler.sample(lf, strategy="block")
result = sampler.sample(lf, strategy="multi_stage")
result = sampler.sample(lf, strategy="column_aware")
result = sampler.sample(lf, strategy="progressive")
```

### Convenience Functions

```python
from truthound.profiler.enterprise_sampling import (
    sample_large_dataset,
    estimate_optimal_sample_size,
    classify_dataset_scale,
)

# Quick sampling
result = sample_large_dataset(
    lf,
    target_rows=50_000,
    quality="standard",
    time_budget_seconds=30.0,
)

# Estimate optimal sample size
size = estimate_optimal_sample_size(
    total_rows=100_000_000,
    confidence_level=0.95,
    margin_of_error=0.05,
)

# Classify dataset scale
scale = classify_dataset_scale(total_rows)  # Returns ScaleCategory enum
```

### Sampling in Validators

Use `EnterpriseScaleSamplingMixin` in custom validators:

```python
from truthound.validators.base import Validator, EnterpriseScaleSamplingMixin

class MyLargeDataValidator(Validator, EnterpriseScaleSamplingMixin):
    sampling_threshold = 10_000_000   # Enable sampling above 10M rows
    sampling_target_rows = 100_000    # Target sample size
    sampling_quality = "standard"

    def validate(self, lf):
        # Automatically samples if dataset is large
        sampled_lf, metrics = self._sample_for_validation(lf)

        # Validate on sampled data
        issues = self._do_validation(sampled_lf)

        # Extrapolate counts if sampled
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
from truthound.profiler import generate_suite, ExportFormat

suite = generate_suite(profile)

# YAML (human-readable)
suite.save("rules.yaml", format=ExportFormat.YAML)

# JSON (machine-readable)
suite.save("rules.json", format=ExportFormat.JSON)

# Python (executable)
suite.save("rules.py", format=ExportFormat.PYTHON)

# TOML (config-friendly)
suite.save("rules.toml", format=ExportFormat.TOML)
```

### Python Export Styles

```python
# Functional style
suite.save("rules.py", format=ExportFormat.PYTHON, style="functional")

# Class-based style
suite.save("rules.py", format=ExportFormat.PYTHON, style="class_based")

# Declarative style
suite.save("rules.py", format=ExportFormat.PYTHON, style="declarative")
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

## Configuration Reference

### ProfilerConfig

```python
@dataclass
class ProfilerConfig:
    # Sampling
    sample_size: int = 50000
    sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE

    # Timeout
    timeout_seconds: int = 120
    timeout_per_column: int = 30

    # Pattern detection
    enable_pattern_detection: bool = True
    pattern_sample_size: int = 1000
    min_pattern_confidence: float = 0.8

    # Caching
    enable_caching: bool = True
    cache_dir: str = ".truthound/cache"
    cache_ttl_hours: int = 24

    # Output
    strictness: Strictness = Strictness.MEDIUM
    include_samples: bool = True
    max_samples: int = 5

    # Advanced
    enable_correlations: bool = False
    correlation_threshold: float = 0.7
    detect_duplicates: bool = True
    max_cardinality: int = 1000
```

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
# Profile a file
profile = profile_file(
    path: str,
    config: ProfilerConfig | None = None,
) -> TableProfile

# Profile a DataFrame/LazyFrame
profile = profile_data(
    data: pl.DataFrame | pl.LazyFrame,
    config: ProfilerConfig | None = None,
) -> TableProfile

# Generate validation suite
suite = generate_suite(
    profile: TableProfile,
    strictness: Strictness = Strictness.MEDIUM,
    preset: str = "default",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> ValidationSuite

# Quick suite (profile + generate)
suite = quick_suite(
    path: str,
    output: str,
    format: str = "yaml",
    strictness: str = "medium",
) -> ValidationSuite

# Save/load profiles
save_profile(profile: TableProfile, path: str) -> None
profile = load_profile(path: str) -> TableProfile
```

### Data Types

```python
class DataType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    UUID = "uuid"
    URL = "url"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    KOREAN_RRN = "korean_rrn"
    KOREAN_PHONE = "korean_phone"
    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    FREETEXT = "freetext"
    JSON = "json"
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
    enable_pattern_detection=True,
    strictness="medium",
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
suite = generate_suite(profile)
suite.save("sales_rules.yaml")
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

- [Validators Reference](VALIDATORS.md)
- [Statistical Methods](STATISTICAL_METHODS.md)
- [Checkpoint & CI/CD](CHECKPOINT.md)
- [Storage Backends](STORES.md)
- [Examples](EXAMPLES.md)
