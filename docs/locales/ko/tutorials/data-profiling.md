# 데이터 프로파일링 Tutorial

튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

튜토리얼에서 Truthound을(를) 다루는 항목입니다:

- Understanding 데이터 품질 before 검증
- 튜토리얼에서 Generating을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 Auto-generating을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Prerequisites

- 튜토리얼에서 Truthound, `pip install truthound`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 JSON, Sample, CSV, Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic 프로파일링

### Using the API

튜토리얼에서 `th.profile()`, `ProfileReport`, ProfileReport을(를) 다루는 항목입니다:

```python
import truthound as th

# Profile a file - returns ProfileReport
profile = th.profile("data.csv")

# View summary
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
print(f"Size: {profile.size_bytes / 1024:.2f} KB")

# Column details (columns is a list of dicts)
# Note: Both 'null_pct'/'unique_pct' and 'null_count'/'unique_count' are available
for col in profile.columns:
    print(f"\n{col['name']} ({col['dtype']}):")
    print(f"  Null %: {col['null_pct']}")       # or col['null_count'] for count
    print(f"  Unique %: {col['unique_pct']}")   # or col['unique_count'] for count
    if col.get('min'):
        print(f"  Range: [{col['min']}, {col['max']}]")

# Print formatted report
profile.print()
```

### Using the CLI

```bash
# Basic profile
truthound profile data.csv

# Output as JSON
truthound profile data.csv --format json > profile.json

# Auto-profile with rule generation
truthound auto-profile data.csv -o profile.json
```

## Advanced 프로파일링 with Data프로파일러

튜토리얼에서 `DataProfiler`, DataProfiler을(를) 다루는 항목입니다:

```python
from truthound.profiler.table_profiler import DataProfiler
from truthound.profiler.base import ProfilerConfig
import polars as pl

# Configure profiler
config = ProfilerConfig(
    sample_size=10000,         # Sample for large datasets
    include_patterns=True,     # Detect patterns (email, phone, etc.)
    include_correlations=True, # Calculate correlations
    n_jobs=4,                  # Parallel processing threads
)

# Create profiler
profiler = DataProfiler(config=config)

# Profile data - returns TableProfile
df = pl.read_parquet("data.parquet")
table_profile = profiler.profile(df.lazy(), name="my_data")

# Access results
print(f"Row count: {table_profile.row_count}")
print(f"Duplicate rows: {table_profile.duplicate_row_count}")
print(f"Duration: {table_profile.profile_duration_ms:.2f}ms")

# Column profiles (TableProfile.columns is a tuple of ColumnProfile)
for col_profile in table_profile.columns:
    print(f"\n{col_profile.name}:")
    print(f"  Physical type: {col_profile.physical_type}")
    print(f"  Inferred type: {col_profile.inferred_type.value}")
    print(f"  Nulls: {col_profile.null_count} ({col_profile.null_ratio:.2%})")
    print(f"  Unique: {col_profile.distinct_count} ({col_profile.unique_ratio:.2%})")

    # Distribution stats for numeric columns
    if col_profile.distribution:
        dist = col_profile.distribution
        print(f"  Mean: {dist.mean:.2f}, Std: {dist.std:.2f}")
        print(f"  Range: [{dist.min}, {dist.max}]")

    # Detected patterns (email, phone, URL, etc.)
    if col_profile.detected_patterns:
        patterns = [p.pattern for p in col_profile.detected_patterns]
        print(f"  Detected patterns: {patterns}")
```

### Convenience Functions

```python
from truthound.profiler.table_profiler import profile_file, profile_dataframe

# Profile from file - returns TableProfile
profile = profile_file("data.parquet")

# Profile DataFrame
import polars as pl
df = pl.read_csv("data.csv")
profile = profile_dataframe(df, name="my_data")

# Convert to dict for serialization
profile_dict = profile.to_dict()
```

### Specialized 테이블 Analyzers

튜토리얼에서 `DataProfiler`, DataProfiler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

#### DuplicateRowAnalyzer

튜토리얼에서 Identifies을(를) 다루는 항목입니다:

```python
from truthound.profiler import DuplicateRowAnalyzer, ProfilerConfig

analyzer = DuplicateRowAnalyzer()
result = analyzer.analyze(df.lazy(), ProfilerConfig())

print(f"Duplicate rows: {result['duplicate_row_count']}")
print(f"Duplicate ratio: {result['duplicate_row_ratio']:.2%}")
```

#### MemoryEstimator

튜토리얼에서 Estimates을(를) 다루는 항목입니다:

```python
from truthound.profiler import MemoryEstimator, ProfilerConfig

estimator = MemoryEstimator()
result = estimator.analyze(df.lazy(), ProfilerConfig())

size_mb = result["estimated_memory_bytes"] / (1024 * 1024)
print(f"Estimated memory: {size_mb:.2f} MB")
```

#### CorrelationAnalyzer

튜토리얼에서 Computes을(를) 다루는 항목입니다:

```python
from truthound.profiler import CorrelationAnalyzer, ProfilerConfig

# Configure correlation threshold
analyzer = CorrelationAnalyzer(threshold=0.5)
config = ProfilerConfig(correlation_threshold=0.5)
result = analyzer.analyze(df.lazy(), config)

for col1, col2, corr in result["correlations"]:
    direction = "positive" if corr > 0 else "negative"
    print(f"{col1} <-> {col2}: {corr:.3f} ({direction} correlation)")
```

#### Custom TableAnalyzer Implementation

튜토리얼에서 `TableAnalyzer`, TableAnalyzer을(를) 다루는 항목입니다:

```python
from truthound.profiler import TableAnalyzer, ProfilerConfig
import polars as pl

class RowCountAnalyzer(TableAnalyzer):
    """Custom analyzer for row count categorization."""
    name = "row_count_custom"

    def analyze(self, lf: pl.LazyFrame, config: ProfilerConfig) -> dict:
        row_count = lf.select(pl.len()).collect().item()
        return {
            "custom_row_count": row_count,
            "row_category": (
                "small" if row_count < 100
                else "medium" if row_count < 1000
                else "large"
            ),
        }

# Register custom analyzer with DataProfiler
profiler = DataProfiler()
profiler.add_table_analyzer(RowCountAnalyzer())
profile = profiler.profile(df.lazy(), name="custom_analysis")
```

## Generating 검증 Rules

### From 프로파일 to Rules

```bash
# Generate validation suite from profile
truthound generate-suite profile.json -o rules.yaml

# One-step: profile + generate suite
truthound quick-suite data.csv -o rules.yaml

# With specific categories
truthound quick-suite data.csv -o rules.yaml --categories completeness,uniqueness,range
```

### Using the API

```python
from truthound.profiler.suite_export import SuiteExporter
from truthound.profiler.table_profiler import profile_file

# Profile data
profile = profile_file("data.csv")

# Export as validation suite
exporter = SuiteExporter()
suite = exporter.export(profile)

# Save suite
suite.save("validation_suite.yaml")

# Use for validation
import truthound as th
report = th.check("new_data.csv", schema="validation_suite.yaml")
```

## 스키마 Learning

### Auto-Learn 스키마 with Constraints

```python
import truthound as th

# Learn schema with constraint inference
schema = th.learn(
    "baseline.csv",
    infer_constraints=True,
    categorical_threshold=20  # Max unique values for categorical
)

# View inferred constraints
for col in schema.columns.values():
    print(f"{col.name}:")
    print(f"  Type: {col.dtype}")
    print(f"  Nullable: {col.nullable}")
    if col.min_value is not None:
        print(f"  Range: [{col.min_value}, {col.max_value}]")
    if col.allowed_values:
        print(f"  Allowed: {col.allowed_values}")

# Save schema
schema.save("schema.yaml")

# Validate new data against schema
report = th.check("new_data.csv", schema=schema)
```

### Zero-설정 with Auto 캐싱

```python
import truthound as th

# First run: learns and caches schema
report = th.check("data.csv", auto_schema=True)

# Subsequent runs: uses cached schema
report = th.check("data.csv", auto_schema=True)

# Cache is invalidated when file changes (based on fingerprint)
```

## Data 드리프트 Detection

튜토리얼에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
튜토리얼에서 `compare()`, `truthound.drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
튜토리얼에서 `ProfileComparator`, ProfileComparator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### High-Level API: `truthound.drift.compare()`

튜토리얼에서 `compare()`을(를) 다루는 항목입니다:

```python
from truthound.drift import compare

# Compare baseline and current data
drift = compare("train.csv", "production.csv")
print(drift)

if drift.has_drift:
    print("Data drift detected!")
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"  - {col_drift.column}: {col_drift.result.method} = {col_drift.result.statistic:.4f}")

# Check for high drift
if drift.has_high_drift:
    print("WARNING: High drift detected!")

# Get list of drifted column names
drifted_cols = drift.get_drifted_columns()
print(f"Drifted columns: {drifted_cols}")
```

### Advanced API: ProfileComparator

튜토리얼에서 `ProfileComparator`, `TableProfile`, ProfileComparator, TableProfile을(를) 다루는 항목입니다:

```python
from truthound.profiler import (
    DataProfiler,
    ProfileComparator,
    compare_profiles,
    DriftType,
    DriftSeverity,
    DriftThresholds,
)

# Generate profiles
profiler = DataProfiler()
baseline_profile = profiler.profile(baseline_df.lazy(), name="baseline")
current_profile = profiler.profile(current_df.lazy(), name="current")

# Compare using ProfileComparator
comparator = ProfileComparator()
comparison = comparator.compare(baseline_profile, current_profile)

print(f"Has Drift: {comparison.has_drift}")
print(f"Total Drifts: {comparison.drift_count}")

# Alternatively, use the convenience function
comparison = compare_profiles(baseline_profile, current_profile)
```

#### Filtering by 드리프트 Type and Severity

```python
# Filter by drift type
completeness_drifts = comparison.get_by_type(DriftType.COMPLETENESS)
distribution_drifts = comparison.get_by_type(DriftType.DISTRIBUTION)
range_drifts = comparison.get_by_type(DriftType.RANGE)
cardinality_drifts = comparison.get_by_type(DriftType.CARDINALITY)

# Filter by severity
critical_drifts = comparison.get_by_severity(DriftSeverity.CRITICAL)
warning_drifts = comparison.get_by_severity(DriftSeverity.WARNING)
info_drifts = comparison.get_by_severity(DriftSeverity.INFO)

# Get specific column comparison
age_comparison = comparison.get_column("age")
if age_comparison and age_comparison.has_drift:
    for drift in age_comparison.drifts:
        print(f"  {drift.drift_type}: {drift.severity}")
```

#### Custom 드리프트 Thresholds

```python
# Configure sensitive thresholds
sensitive_thresholds = DriftThresholds(
    null_ratio_warning=0.01,   # 1% change triggers warning
    null_ratio_critical=0.05,
    mean_warning=0.05,
    mean_critical=0.1,
)

# Configure lenient thresholds
lenient_thresholds = DriftThresholds(
    null_ratio_warning=0.2,    # 20% change required for warning
    null_ratio_critical=0.5,
    mean_warning=0.3,
    mean_critical=0.5,
)

# Apply thresholds to comparator
comparator_sensitive = ProfileComparator(thresholds=sensitive_thresholds)
comparison = comparator_sensitive.compare(baseline_profile, current_profile)
```

#### Generating 드리프트 리포트

```python
# Generate text report
report = comparison.to_report()
print(report)

# Output includes:
# - Summary with drift counts by severity
# - Detailed breakdown of critical and warning drifts
# - Per-column change descriptions
```

### Specifying Detection Method

```python
from truthound.drift import compare

# Auto-select based on data type (default, recommended)
drift = compare(baseline, current, method="auto")

# Kolmogorov-Smirnov test (numeric columns only)
drift = compare(baseline, current, method="ks")

# Population Stability Index (numeric columns only)
drift = compare(baseline, current, method="psi")

# Chi-square test (categorical columns)
drift = compare(baseline, current, method="chi2")

# Jensen-Shannon divergence (works with any column type)
drift = compare(baseline, current, method="js")

# Custom threshold
drift = compare(baseline, current, threshold=0.2)

# With sampling for large datasets
drift = compare(baseline, current, sample_size=10000)
```

> 튜토리얼에서 `ks`, `psi`, Note을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 튜토리얼에서 `--columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 튜토리얼에서 `method="auto"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 
> 튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 튜토리얼에서 Compare, PSI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

`method="auto"`에서는 중첩 `Struct`, `List` 열을 결정론적 JSON 범주로 변환해
비교합니다. 객체의 key 순서만 달라진 경우에는 drift로 오판하지 않고, 중첩 값의
실제 변경은 비교 결과에 반영합니다. `ColumnDrift.dtype`에는 원래 Polars dtype을
유지합니다. `ks`, `psi` 같은 명시적 수치 방식에는 여전히 수치 열만 전달해야
합니다.

## 권장 방식

### 1. 프로파일 Before 검증

튜토리얼에서 Always을(를) 다루는 항목입니다:

```python
import truthound as th
from truthound.profiler.table_profiler import profile_file

# Profile first to understand the data
profile = profile_file("new_dataset.csv")
print(f"Rows: {profile.row_count}, Columns: {profile.column_count}")

# Then set up appropriate validation
schema = th.learn("new_dataset.csv", infer_constraints=True)
schema.save("new_dataset_schema.yaml")
```

### 튜토리얼 개요

```python
from truthound.profiler.table_profiler import DataProfiler
from truthound.profiler.base import ProfilerConfig

config = ProfilerConfig(
    sample_size=50_000,  # Profile 50K rows
    random_seed=42,       # Reproducible sampling
)
profiler = DataProfiler(config=config)
```

### 튜토리얼 개요

```python
import json
from datetime import datetime
from truthound.profiler.table_profiler import profile_file

# Profile and save with timestamp
profile = profile_file("data.csv")
profile_dict = profile.to_dict()

filename = f"profiles/data_{datetime.now():%Y%m%d_%H%M%S}.json"
with open(filename, "w") as f:
    json.dump(profile_dict, f, indent=2, default=str)
```

### 4. Parallel Processing

튜토리얼에서 Enable을(를) 다루는 항목입니다:

```python
from truthound.profiler.table_profiler import DataProfiler
from truthound.profiler.base import ProfilerConfig

config = ProfilerConfig(
    n_jobs=4,  # Use 4 threads for parallel column profiling
)
profiler = DataProfiler(config=config)
```

## Data Structures 레퍼런스

### ProfileReport (from th.프로파일)

The simple 프로파일 리포트 returned by `th.profile()`:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 소스 파일 or data name |
| 튜토리얼에서 `row_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `column_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Number of 컬럼 |
| 튜토리얼에서 `size_bytes`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Estimated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `list[dict]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 summary dicts with `name`, `dtype`, `null_pct`, `unique_pct`, `min`, `max` |

튜토리얼에서 Methods을(를) 다루는 항목입니다:
- `print()` - Print formatted 리포트 to console
- 튜토리얼에서 `to_dict()`, Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 JSON, `to_json()`, Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### TableProfile (from Data프로파일러)

The detailed 프로파일 returned by `DataProfiler.profile()`:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 테이블/dataset name |
| 튜토리얼에서 `row_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `column_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Number of 컬럼 |
| 튜토리얼에서 `estimated_memory_bytes`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `tuple[ColumnProfile]`, ColumnProfile을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Detailed 컬럼 profiles |
| 튜토리얼에서 `duplicate_row_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `duplicate_row_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Duplicate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `correlations`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `tuple`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 correlation pairs |
| 튜토리얼에서 `profile_duration_ms`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 time |

### ColumnProfile

Detailed 프로파일 for a single 컬럼:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name |
| 튜토리얼에서 `physical_type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Polars을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `inferred_type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `DataType`, DataType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Semantic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `null_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `null_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `distinct_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `unique_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Uniqueness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `is_unique`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `is_constant`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `distribution`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `DistributionStats`, DistributionStats을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `top_values`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `tuple[ValueFrequency]`, ValueFrequency을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Most을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `min_length`, `max_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 String을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `detected_patterns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `tuple[PatternMatch]`, PatternMatch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Detected을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `suggested_validators`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `tuple[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Recommended 검증기 |

### 프로파일러Config

설정 options for 프로파일링:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 튜토리얼에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Rows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `random_seed`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `42`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Sampling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `include_patterns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Detect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `include_correlations`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Calculate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `include_distributions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Calculate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `top_n_values`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `10`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Top을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `n_jobs`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `correlation_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `0.7`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Minimum correlation to 리포트 |

### TableAnalyzer Protocol

튜토리얼에서 Base을(를) 다루는 항목입니다:

| 튜토리얼에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Parameters을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Return을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------------|--------|-------------|
| 튜토리얼에서 `analyze`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `lf: LazyFrame, config: ProfilerConfig`, LazyFrame, ProfilerConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Execute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### DriftReport (from `truthound.drift.compare`)

The 드리프트 리포트 returned by `compare()`:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `baseline_source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Baseline data 소스 name |
| 튜토리얼에서 `current_source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Current data 소스 name |
| 튜토리얼에서 `baseline_rows`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `current_rows`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `list[ColumnDrift]`, ColumnDrift을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per-컬럼 드리프트 결과 |

튜토리얼에서 Properties을(를) 다루는 항목입니다:
- `has_drift` - True if any 컬럼 has 드리프트
- 튜토리얼에서 `has_high_drift`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

튜토리얼에서 Methods을(를) 다루는 항목입니다:
- `print()` - Print formatted 리포트 to console
- 튜토리얼에서 `to_dict()`, Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 JSON, `to_json()`, Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 `get_drifted_columns()`, Get을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### ProfileComparison (from ProfileComparator)

튜토리얼에서 `ProfileComparator.compare()`, ProfileComparator.compare을(를) 다루는 항목입니다:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `has_drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Whether을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `drift_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `list[ColumnComparison]`, ColumnComparison을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per-컬럼 comparison 결과 |
| 튜토리얼에서 `all_drifts`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `list[Drift]`, Drift을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Flattened을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

튜토리얼에서 Methods을(를) 다루는 항목입니다:
- 튜토리얼에서 `get_by_type(drift_type)`, `DriftType`, Filter, DriftType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 `get_by_severity(severity)`, `DriftSeverity`, Filter, DriftSeverity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 `get_column(name)`, Get을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- `to_report()` - Generate formatted text 리포트

### DriftType Enumeration

| 튜토리얼에서 Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 튜토리얼에서 `COMPLETENESS`, COMPLETENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `DISTRIBUTION`, DISTRIBUTION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `RANGE`, RANGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `CARDINALITY`, CARDINALITY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `UNIQUENESS`, UNIQUENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### DriftSeverity Enumeration

| 튜토리얼에서 Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 튜토리얼에서 `INFO`, INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Minor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `WARNING`, WARNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Moderate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Significant을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### DriftThresholds

설정 for 드리프트 detection sensitivity:

| 튜토리얼에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 튜토리얼에서 `null_ratio_warning`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `0.05`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `null_ratio_critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `0.1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `mean_warning`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `0.1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Mean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `mean_critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `0.2`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Mean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## CLI Commands 레퍼런스

| 튜토리얼에서 Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|
| 튜토리얼에서 `truthound profile <file>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Basic 프로파일 |
| 튜토리얼에서 `truthound auto-profile <file>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일 with pattern detection |
| 튜토리얼에서 `truthound generate-suite <profile>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Generate rules from 프로파일 |
| 튜토리얼에서 `truthound quick-suite <file>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | One-step 프로파일 + rules |
| 튜토리얼에서 `truthound compare <baseline> <current>`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Compare datasets for 드리프트 |

## API Summary

튜토리얼에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

| 튜토리얼에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 API, High-Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 API, Advanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------------|--------------|
| Basic 프로파일링 | 튜토리얼에서 `th.profile()`, `ProfileReport`, ProfileReport을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `DataProfiler.profile()`, `TableProfile`, DataProfiler.profile, TableProfile을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 드리프트 detection | 튜토리얼에서 `truthound.drift.compare()`, `DriftReport`, DriftReport을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `ProfileComparator.compare()`, `ProfileComparison`, ProfileComparator.compare, ProfileComparison을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 Convenience을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `compare_profiles()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

튜토리얼에서 API을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 튜토리얼에서 Custom, Validator, Tutorial, Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [엔터프라이즈 설정](enterprise-setup.md) - CI/CD 통합 with 프로파일링
- 튜토리얼에서 Profiler, Configuration, Advanced을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 API, Examples, More을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
