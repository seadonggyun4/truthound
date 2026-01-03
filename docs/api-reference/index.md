# API Reference

Complete API documentation for Truthound.

## Core Module

The main entry point for Truthound functionality.

```python
import truthound as th

# Core functions
report = th.check("data.csv")      # Validate data
profile = th.profile("data.csv")   # Profile data
schema = th.learn("data.csv")      # Learn schema
pii = th.scan("data.csv")          # Scan for PII
masked = th.mask("data.csv")       # Mask data
drift = th.compare(old, new)       # Detect drift
```

## Module Index

### Core API

| Module | Description |
|--------|-------------|
| [`truthound.api`](api.md) | Main API functions (check, scan, mask, profile) |
| [`truthound.schema`](schema.md) | Schema learning and validation |
| [`truthound.drift`](drift.md) | Data drift detection |
| [`truthound.report`](report.md) | Report classes and formatting |

### Validation

| Module | Description |
|--------|-------------|
| [`truthound.validators`](validators/index.md) | 289 built-in validators |
| [`truthound.validators.sdk`](validators/sdk/index.md) | Custom validator SDK |

### Profiling

| Module | Description |
|--------|-------------|
| [`truthound.profiler`](profiler/index.md) | Data profiling and rule generation |

### Storage & Reporting

| Module | Description |
|--------|-------------|
| [`truthound.stores`](stores/index.md) | Result storage backends |
| [`truthound.datadocs`](datadocs/index.md) | HTML reports and documentation |

### CI/CD

| Module | Description |
|--------|-------------|
| [`truthound.checkpoint`](checkpoint/index.md) | Checkpoint and CI/CD integration |

### Advanced Features

| Module | Description |
|--------|-------------|
| [`truthound.ml`](ml/index.md) | Machine learning features |
| [`truthound.lineage`](lineage/index.md) | Data lineage tracking |
| [`truthound.realtime`](realtime/index.md) | Streaming validation |

### Extensions

| Module | Description |
|--------|-------------|
| [`truthound.plugins`](plugins/index.md) | Plugin system |

## Quick Examples

### Basic Validation

```python
import truthound as th

# Check with default validators
report = th.check("data.csv")
print(f"Issues found: {report.issue_count}")

# Check with specific validators
report = th.check(
    "data.csv",
    validators=["null", "duplicate", "range"],
    min_severity="high"
)

# Check with schema
schema = th.learn("reference_data.csv")
report = th.check("new_data.csv", schema=schema)
```

### Profiling

```python
from truthound import DataProfiler, ProfilerConfig

# Configure profiler
config = ProfilerConfig(
    include_patterns=True,
    include_correlations=True,
    sample_size=10000
)

# Profile data
profiler = DataProfiler(config=config)
profile = profiler.profile("data.csv")

# Access profile information
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")

for col in profile.columns:
    print(f"{col.name}: {col.inferred_type}, {col.null_ratio:.1%} nulls")
```

### Custom Validators

```python
from truthound.validators.sdk import validator, ValidationResult

@validator("my_validator", category="custom")
def my_custom_validator(df, column: str, threshold: float = 0.1):
    """Custom validation logic."""
    values = df[column]
    invalid_count = (values < 0).sum()
    invalid_ratio = invalid_count / len(values)

    return ValidationResult(
        passed=invalid_ratio <= threshold,
        message=f"Found {invalid_ratio:.1%} invalid values",
        details={"invalid_count": invalid_count}
    )
```
