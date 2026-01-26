# Data Profiling Tutorial

Learn how to profile your data and generate validation rules automatically.

## Overview

Truthound's profiler analyzes your data to generate statistical summaries and automatically infer validation rules. This is essential for:

- Understanding data quality before validation
- Generating baseline schemas for drift detection
- Auto-generating validation rules from data characteristics

## Prerequisites

- Truthound installed (`pip install truthound`)
- Sample data file (CSV, Parquet, or JSON)

## Basic Profiling

### Using the API

The `th.profile()` function returns a `ProfileReport` with basic statistics:

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

## Advanced Profiling with DataProfiler

For detailed profiling with pattern detection and correlations, use the `DataProfiler` class:

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

### Specialized Table Analyzers

The profiler architecture provides modular analyzers for specific analysis tasks. These components can be utilized independently or in conjunction with `DataProfiler` for fine-grained control over the profiling process.

#### DuplicateRowAnalyzer

Identifies duplicate rows within the dataset:

```python
from truthound.profiler import DuplicateRowAnalyzer, ProfilerConfig

analyzer = DuplicateRowAnalyzer()
result = analyzer.analyze(df.lazy(), ProfilerConfig())

print(f"Duplicate rows: {result['duplicate_row_count']}")
print(f"Duplicate ratio: {result['duplicate_row_ratio']:.2%}")
```

#### MemoryEstimator

Estimates memory consumption for datasets:

```python
from truthound.profiler import MemoryEstimator, ProfilerConfig

estimator = MemoryEstimator()
result = estimator.analyze(df.lazy(), ProfilerConfig())

size_mb = result["estimated_memory_bytes"] / (1024 * 1024)
print(f"Estimated memory: {size_mb:.2f} MB")
```

#### CorrelationAnalyzer

Computes pairwise correlations between numeric columns:

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

The `TableAnalyzer` protocol enables extension of profiling capabilities through custom implementations:

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

## Generating Validation Rules

### From Profile to Rules

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

## Schema Learning

### Auto-Learn Schema with Constraints

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

### Zero-Configuration with Auto Caching

```python
import truthound as th

# First run: learns and caches schema
report = th.check("data.csv", auto_schema=True)

# Subsequent runs: uses cached schema
report = th.check("data.csv", auto_schema=True)

# Cache is invalidated when file changes (based on fingerprint)
```

## Data Drift Detection

Truthound provides multiple approaches for detecting data drift between datasets. The high-level `th.compare()` API offers simplified access, while the `ProfileComparator` class provides comprehensive control over the comparison process.

### High-Level API: th.compare()

The `th.compare()` function provides a streamlined interface for drift detection:

```python
import truthound as th

# Compare baseline and current data
drift = th.compare("train.csv", "production.csv")
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

For fine-grained control over drift detection, the `ProfileComparator` class operates on pre-computed `TableProfile` objects:

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

#### Filtering by Drift Type and Severity

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

#### Custom Drift Thresholds

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

#### Generating Drift Reports

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
import truthound as th

# Auto-select based on data type (default, recommended)
drift = th.compare(baseline, current, method="auto")

# Kolmogorov-Smirnov test (numeric columns only)
drift = th.compare(baseline, current, method="ks")

# Population Stability Index (numeric columns only)
drift = th.compare(baseline, current, method="psi")

# Chi-square test (categorical columns)
drift = th.compare(baseline, current, method="chi2")

# Jensen-Shannon divergence (works with any column type)
drift = th.compare(baseline, current, method="js")

# Custom threshold
drift = th.compare(baseline, current, threshold=0.2)

# With sampling for large datasets
drift = th.compare(baseline, current, sample_size=10000)
```

> **Note:** `ks` and `psi` methods only work with numeric columns. If your data contains
> non-numeric columns (e.g., strings), use `--columns` to specify numeric columns only,
> or use `method="auto"` which automatically selects the appropriate method per column.
>
> ```python
> # Compare only numeric columns with PSI
> drift = th.compare(baseline, current, method="psi", columns=["age", "salary", "score"])
> ```

## Best Practices

### 1. Profile Before Validation

Always profile new data sources before setting up validation:

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

### 2. Use Sampling for Large Datasets

```python
from truthound.profiler.table_profiler import DataProfiler
from truthound.profiler.base import ProfilerConfig

config = ProfilerConfig(
    sample_size=50_000,  # Profile 50K rows
    random_seed=42,       # Reproducible sampling
)
profiler = DataProfiler(config=config)
```

### 3. Store Profiles for Historical Analysis

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

Enable parallel column profiling for faster results:

```python
from truthound.profiler.table_profiler import DataProfiler
from truthound.profiler.base import ProfilerConfig

config = ProfilerConfig(
    n_jobs=4,  # Use 4 threads for parallel column profiling
)
profiler = DataProfiler(config=config)
```

## Data Structures Reference

### ProfileReport (from th.profile)

The simple profile report returned by `th.profile()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `source` | `str` | Source file or data name |
| `row_count` | `int` | Number of rows |
| `column_count` | `int` | Number of columns |
| `size_bytes` | `int` | Estimated size in bytes |
| `columns` | `list[dict]` | Column summary dicts with `name`, `dtype`, `null_pct`, `unique_pct`, `min`, `max` |

Methods:
- `print()` - Print formatted report to console
- `to_dict()` - Convert to dictionary
- `to_json()` - Convert to JSON string

### TableProfile (from DataProfiler)

The detailed profile returned by `DataProfiler.profile()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Table/dataset name |
| `row_count` | `int` | Number of rows |
| `column_count` | `int` | Number of columns |
| `estimated_memory_bytes` | `int` | Memory estimate |
| `columns` | `tuple[ColumnProfile]` | Detailed column profiles |
| `duplicate_row_count` | `int` | Number of duplicate rows |
| `duplicate_row_ratio` | `float` | Duplicate row ratio |
| `correlations` | `tuple` | Column correlation pairs |
| `profile_duration_ms` | `float` | Profiling time |

### ColumnProfile

Detailed profile for a single column:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Column name |
| `physical_type` | `str` | Polars dtype as string |
| `inferred_type` | `DataType` | Semantic type (email, url, phone, etc.) |
| `null_count` | `int` | Null value count |
| `null_ratio` | `float` | Null ratio (0.0-1.0) |
| `distinct_count` | `int` | Unique value count |
| `unique_ratio` | `float` | Uniqueness ratio |
| `is_unique` | `bool` | True if all values unique |
| `is_constant` | `bool` | True if all values same |
| `distribution` | `DistributionStats` | Numeric statistics (mean, std, min, max, etc.) |
| `top_values` | `tuple[ValueFrequency]` | Most frequent values |
| `min_length` / `max_length` | `int` | String length bounds |
| `detected_patterns` | `tuple[PatternMatch]` | Detected data patterns |
| `suggested_validators` | `tuple[str]` | Recommended validators |

### ProfilerConfig

Configuration options for profiling:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_size` | `int \| None` | `None` | Rows to sample |
| `random_seed` | `int` | `42` | Sampling seed |
| `include_patterns` | `bool` | `True` | Detect patterns |
| `include_correlations` | `bool` | `False` | Calculate correlations |
| `include_distributions` | `bool` | `True` | Calculate distribution stats |
| `top_n_values` | `int` | `10` | Top N frequent values |
| `n_jobs` | `int` | `1` | Parallel threads |
| `correlation_threshold` | `float` | `0.7` | Minimum correlation to report |

### TableAnalyzer Protocol

Base protocol for implementing custom table-level analyzers:

| Method | Parameters | Return | Description |
|--------|------------|--------|-------------|
| `analyze` | `lf: LazyFrame, config: ProfilerConfig` | `dict` | Execute analysis and return results |

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the analyzer |

### DriftReport (from th.compare)

The drift report returned by `th.compare()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `baseline_source` | `str` | Baseline data source name |
| `current_source` | `str` | Current data source name |
| `baseline_rows` | `int` | Number of rows in baseline |
| `current_rows` | `int` | Number of rows in current |
| `columns` | `list[ColumnDrift]` | Per-column drift results |

Properties:
- `has_drift` - True if any column has drift
- `has_high_drift` - True if any column has high drift

Methods:
- `print()` - Print formatted report to console
- `to_dict()` - Convert to dictionary
- `to_json()` - Convert to JSON string
- `get_drifted_columns()` - Get list of drifted column names

### ProfileComparison (from ProfileComparator)

The detailed comparison result returned by `ProfileComparator.compare()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `has_drift` | `bool` | Whether any drift was detected |
| `drift_count` | `int` | Total number of detected drifts |
| `columns` | `list[ColumnComparison]` | Per-column comparison results |
| `all_drifts` | `list[Drift]` | Flattened list of all drifts |

Methods:
- `get_by_type(drift_type)` - Filter drifts by `DriftType`
- `get_by_severity(severity)` - Filter drifts by `DriftSeverity`
- `get_column(name)` - Get comparison for specific column
- `to_report()` - Generate formatted text report

### DriftType Enumeration

| Value | Description |
|-------|-------------|
| `COMPLETENESS` | Changes in null ratio |
| `DISTRIBUTION` | Changes in statistical distribution |
| `RANGE` | Changes in min/max values |
| `CARDINALITY` | Changes in distinct value count |
| `UNIQUENESS` | Changes in uniqueness ratio |

### DriftSeverity Enumeration

| Value | Description |
|-------|-------------|
| `INFO` | Minor change, informational only |
| `WARNING` | Moderate change, may require attention |
| `CRITICAL` | Significant change, likely requires action |

### DriftThresholds

Configuration for drift detection sensitivity:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `null_ratio_warning` | `float` | `0.05` | Null ratio change for warning |
| `null_ratio_critical` | `float` | `0.1` | Null ratio change for critical |
| `mean_warning` | `float` | `0.1` | Mean change ratio for warning |
| `mean_critical` | `float` | `0.2` | Mean change ratio for critical |

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `truthound profile <file>` | Basic profile |
| `truthound auto-profile <file>` | Profile with pattern detection |
| `truthound generate-suite <profile>` | Generate rules from profile |
| `truthound quick-suite <file>` | One-step profile + rules |
| `truthound compare <baseline> <current>` | Compare datasets for drift |

## API Summary

The following table summarizes the two complementary approaches for data profiling and comparison:

| Use Case | High-Level API | Advanced API |
|----------|----------------|--------------|
| Basic profiling | `th.profile()` → `ProfileReport` | `DataProfiler.profile()` → `TableProfile` |
| Drift detection | `th.compare()` → `DriftReport` | `ProfileComparator.compare()` → `ProfileComparison` |
| Convenience function | - | `compare_profiles()` |

The high-level API is recommended for standard use cases, while the advanced API provides additional capabilities for custom analysis pipelines, threshold configuration, and integration with scheduling systems.

## Next Steps

- [Custom Validator Tutorial](custom-validator.md) - Create validators from learned patterns
- [Enterprise Setup](enterprise-setup.md) - CI/CD integration with profiling
- [Profiler Configuration](../guides/configuration/profiler-config.md) - Advanced configuration options
- [Examples](examples.md) - More API usage examples
