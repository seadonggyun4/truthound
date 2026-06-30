# 데이터 프로파일링 Guide

실무 운영 가이드에서 Truthound, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 워크플로우s

### 워크플로우 1: 프로파일 and Validate 파이프라인

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

### 워크플로우 2: 스키마 Evolution Detection

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

### 워크플로우 3: Incremental 프로파일링 with 캐시

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

### 워크플로우 4: Large 파일 Streaming 프로파일

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Full Documentation

실무 운영 가이드에서 Truthound, Auto-Profiling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. [빠른 시작](#quick-start)
3. 실무 운영 가이드에서 Core, Components을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 CLI, Commands을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Sampling, Strategies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 Streaming, Pattern, Matching을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
7. 실무 운영 가이드에서 Rule, Generation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
8. [캐싱 & Incremental 프로파일링](#caching--incremental-profiling)
9. [복원력 & Timeout](#resilience--timeout)
10. [관측성](#observability)
11. 실무 운영 가이드에서 Distributed, Processing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
12. [프로파일 Comparison & 드리프트 Detection](#profile-comparison--drift-detection)
13. 실무 운영 가이드에서 Quality, Scoring을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
14. 실무 운영 가이드에서 ML-based, Type, Inference을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
15. 실무 운영 가이드에서 Automatic, Threshold, Tuning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
16. [프로파일 Visualization](#profile-visualization)
17. 실무 운영 가이드에서 Internationalization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
18. [Incremental 검증](#incremental-validation)
19. [설정 레퍼런스](#configuration-reference)
20. [API 레퍼런스](#api-reference)

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Auto-Profiling을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Automatic, Data, Profiling, Column을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Rule, Generation, Generate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 JSON, YAML, Validation, Suite, Export, Python, TOML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Streaming, Support, Process을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Memory, Safety, Sampling, OOM, LazyFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Process, Isolation, Timeout을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Schema, Versioning, Forward/backward을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Caching, Incremental을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Distributed, Processing, Multi-threaded, Spark, Dask, Ray을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 아키텍처

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Core Components

### Data프로파일러

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### Column프로파일러

실무 운영 가이드에서 Analyzes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Contains을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 `TableProfile`, `frozen=True`, `to_dict()`, `json.dumps()`, Note, TableProfile, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CLI Commands

### auto-프로파일

실무 운영 가이드에서 Profile을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Generate 검증 rules from a 프로파일.

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

실무 운영 가이드에서 One-step을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Sampling Strategies

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Available Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|----------|
| 실무 운영 가이드에서 `NONE`, NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Small을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HEAD`, HEAD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Take을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RANDOM`, RANDOM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Random을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SYSTEMATIC`, SYSTEMATIC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Nth을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ordered을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STRATIFIED`, STRATIFIED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maintain을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Categorical 컬럼 |
| 실무 운영 가이드에서 `RESERVOIR`, RESERVOIR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Reservoir을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Streaming을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ADAPTIVE`, ADAPTIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Auto-select을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HASH`, HASH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Reproducibility을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 `.head(limit).collect()`, OOM을(를) 다루는 항목입니다:

```python
# Internal implementation ensures memory safety
def sample(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    # Never calls .collect() on full dataset
    # Uses .head(limit) before collection
    return lf.head(self.config.max_rows)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Enterprise-Scale Sampling (100M+ rows)

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

> 실무 운영 가이드에서 Full, Documentation, See, Enterprise, Sampling, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 빠른 시작

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

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Row, Count을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-----------|----------|--------|
| 실무 운영 가이드에서 SMALL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Full을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼-aware | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 LARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Block을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 XLARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multi-stage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 XXLARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sketches을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Parallel Block Processing

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|----------|
| 실무 운영 가이드에서 `block`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `multi_stage`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hierarchical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `column_aware`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Mixed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type-weighted을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `progressive`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exploratory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Early을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `parallel_block`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Work을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```python
from truthound.profiler.enterprise_sampling import EnterpriseScaleSampler

sampler = EnterpriseScaleSampler(config)

# Auto-select best strategy
result = sampler.sample(lf)

# Or specify explicitly
result = sampler.sample(lf, strategy="block")
result = sampler.sample(lf, strategy="multi_stage")
```

### 검증기 with Large Data Support

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Streaming Pattern Matching

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Aggregation Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `INCREMENTAL`, INCREMENTAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Running을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WEIGHTED`, WEIGHTED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Size-weighted을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SLIDING_WINDOW`, SLIDING_WINDOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Recent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `EXPONENTIAL`, EXPONENTIAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exponential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CONSENSUS`, CONSENSUS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Agreement을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ADAPTIVE`, ADAPTIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Auto-select을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Rule Generation

### Rule Categories

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Example, Rules을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|---------------|
| 실무 운영 가이드에서 `schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 structure | 컬럼 exists, type check |
| 실무 운영 가이드에서 `stats`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `pattern`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Format 검증 | 실무 운영 가이드에서 Email, UUID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `completeness`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `uniqueness`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Duplicate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Primary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `distribution`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Allowed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Strictness Levels

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `loose`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Permissive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `medium`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Balanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `strict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tight을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Presets

| 실무 운영 가이드에서 Preset을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|----------|
| 실무 운영 가이드에서 `default`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `strict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Production을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `loose`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Development/testing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `minimal`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Essential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `comprehensive`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ci_cd`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Optimized for CI/CD 파이프라인 |
| 실무 운영 가이드에서 `schema_only`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Structure 검증 only |
| 실무 운영 가이드에서 `format_only`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Format/pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 캐싱 & Incremental 프로파일링

### Fingerprint-Based 캐싱

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

### Incremental 프로파일링

실무 운영 가이드에서 Only을(를) 다루는 항목입니다:

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

### 캐시 Backends

| 실무 운영 가이드에서 Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|
| 실무 운영 가이드에서 `memory`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | In-memory LRU 캐시 |
| 실무 운영 가이드에서 `file`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Disk-based JSON 캐시 |
| 실무 운영 가이드에서 `redis`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Redis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```python
from truthound.profiler import CacheBackend, RedisCacheBackend

# Redis cache with fallback
cache = RedisCacheBackend(
    host="redis.example.com",
    fallback=FileCacheBackend(".truthound/cache"),
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 복원력 & Timeout

### Process Isolation

실무 운영 가이드에서 Polars, Python을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 Prevent을(를) 다루는 항목입니다:

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

### Resilient 캐시

실무 운영 가이드에서 Automatic을(를) 다루는 항목입니다:

```python
from truthound.profiler import ResilientCacheBackend, FallbackChain

cache = ResilientCacheBackend(
    primary=RedisCacheBackend(host="redis"),
    fallback=FileCacheBackend(".cache"),
    circuit_breaker=CircuitBreakerConfig(failure_threshold=3),
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 관측성

### OpenTelemetry 통합

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

### 메트릭 Collection

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Distributed Processing

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Available Backends

| 실무 운영 가이드에서 Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution, Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Status을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|-------------------|--------|
| 실무 운영 가이드에서 `local`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multi-threaded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ThreadPoolExecutor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ready을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `process`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multi-process을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ProcessPoolExecutor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ready을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `spark`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Apache, Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SparkContext-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Framework을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `dask`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Delayed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Framework을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ray`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ray을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Actor-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Framework을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Optimal, Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|------------------|
| 실무 운영 가이드에서 `row_based`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Split을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Large을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `column_based`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일 컬럼 in parallel | Many 컬럼, limited rows |
| 실무 운영 가이드에서 `hybrid`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Combine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Balanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `hash`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hash-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Reproducible 결과 |

### Execution Backend Selection

실무 운영 가이드에서 `AdaptiveStrategy`, AdaptiveStrategy을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 프로파일 Comparison & 드리프트 Detection

실무 운영 가이드에서 Compare을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 드리프트 Types

| 드리프트 Type | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------|
| 실무 운영 가이드에서 `COMPLETENESS`, COMPLETENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `UNIQUENESS`, UNIQUENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DISTRIBUTION`, DISTRIBUTION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RANGE`, RANGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CARDINALITY`, CARDINALITY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 드리프트 Severity

| 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `LOW`, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Minor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MEDIUM`, MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Notable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HIGH`, HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Significant을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Breaking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

### Individual 드리프트 Detectors

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Quality Scoring

실무 운영 가이드에서 Evaluate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Quality Levels

| 실무 운영 가이드에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Score, Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|-------------|
| 실무 운영 가이드에서 `EXCELLENT`, EXCELLENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `GOOD`, GOOD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Balanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FAIR`, FAIR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Acceptable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `POOR`, POOR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Needs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ML-based Type Inference

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Inference Models

| 실무 운영 가이드에서 Model을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|----------|
| 실무 운영 가이드에서 `RuleBasedModel`, RuleBasedModel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NaiveBayesModel`, NaiveBayesModel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Probabilistic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Good을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `EnsembleModel`, EnsembleModel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Combines을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Automatic Threshold Tuning

실무 운영 가이드에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Tuning Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `CONSERVATIVE`, CONSERVATIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Strict을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BALANCED`, BALANCED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Balance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PERMISSIVE`, PERMISSIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Relaxed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ADAPTIVE`, ADAPTIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Learns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STATISTICAL`, STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DOMAIN_AWARE`, DOMAIN_AWARE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Incorporates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 프로파일 Visualization

Generate HTML 리포트 from 프로파일 결과.

### Chart Types

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `BAR`, BAR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HISTOGRAM`, HISTOGRAM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Distribution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PIE`, PIE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Pie을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `LINE`, LINE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Trend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HEATMAP`, HEATMAP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Correlation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Themes

| 실무 운영 가이드에서 Theme을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `LIGHT`, LIGHT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Light을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DARK`, DARK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CORPORATE`, CORPORATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Professional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MINIMAL`, MINIMAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Clean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLORFUL`, COLORFUL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Vibrant을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

### Compare Profiles 리포트

```python
from truthound.profiler import compare_profile_reports

# Generate comparison report
html = compare_profile_reports(profile_old, profile_new)
with open("comparison_report.html", "w") as f:
    f.write(html)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Internationalization (i18n)

실무 운영 가이드에서 Multi-language을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Supported Locales

| 실무 운영 가이드에서 Locale을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Language을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|----------|
| 실무 운영 가이드에서 `en`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 English을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ko`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Korean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ja`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Japanese을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `zh`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Chinese을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `de`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 German을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fr`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 French을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `es`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Spanish을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Incremental 검증

실무 운영 가이드에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 검증기

| 검증기 | 실무 운영 가이드에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|
| 실무 운영 가이드에서 `ChangeDetectionAccuracyValidator`, ChangeDetectionAccuracyValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Validates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SchemaChangeValidator`, SchemaChangeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validates 스키마 change detection |
| 실무 운영 가이드에서 `StalenessValidator`, StalenessValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validates 프로파일 freshness |
| 실무 운영 가이드에서 `FingerprintConsistencyValidator`, FingerprintConsistencyValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Validates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FingerprintSensitivityValidator`, FingerprintSensitivityValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Validates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ProfileMergeValidator`, ProfileMergeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validates 프로파일 merging |
| 실무 운영 가이드에서 `DataIntegrityValidator`, DataIntegrityValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Validates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PerformanceValidator`, PerformanceValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validates 성능 metrics |

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

### 사용자 정의 검증기s

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

### 검증 Runner

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 레퍼런스

### 프로파일러Config

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

실무 운영 가이드에서 `ProcessTimeoutExecutor`, `ProfileCache`, Note, Timeout, ProcessTimeoutExecutor, ProfileCache을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 환경 변수

| 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|---------|
| 실무 운영 가이드에서 `TRUTHOUND_CACHE_DIR`, TRUTHOUND_CACHE_DIR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 directory | 실무 운영 가이드에서 `.truthound/cache`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRUTHOUND_SAMPLE_SIZE`, TRUTHOUND_SAMPLE_SIZE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `50000`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRUTHOUND_TIMEOUT`, TRUTHOUND_TIMEOUT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `120`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRUTHOUND_LOG_LEVEL`, TRUTHOUND_LOG_LEVEL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 로깅 level | 실무 운영 가이드에서 `INFO`, INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 레퍼런스

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

실무 운영 가이드에서 `quick_suite()`, `profile_file()`, `generate_suite()`, `save_suite()`, Note, There을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시

### Basic 프로파일링 파이프라인

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

### Streaming Large 파일

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

### CI/CD 통합

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 스키마 버전 관리

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Version Format

```
major.minor.patch
```

- 실무 운영 가이드에서 Breaking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 New을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Bug을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Automatic 마이그레이션

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

### 검증

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 Schema, Evolution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Drift, Detection, Detect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Quality, Scoring, Evaluate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [검증기 Guide](../validators.md)
- 실무 운영 가이드에서 Statistical, Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [체크포인트 & CI/CD](../checkpoints.md)
- 실무 운영 가이드에서 Storage, Backends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [예시](../../tutorials/examples.md)
