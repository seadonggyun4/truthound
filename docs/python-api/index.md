# Python API Reference

Complete reference for the Truthound Python API.

## Installation

```bash
pip install truthound
```

## Quick Start

```python
import truthound as th

# Validate data
report = th.check("data.csv")

# Learn schema
schema = th.learn("baseline.csv")

# Scan for PII
pii_report = th.scan("customers.csv")

# Mask sensitive data
masked_df = th.mask(df, strategy="hash")

# Profile data
profile = th.profile("data.csv")

# Compare datasets (drift detection)
drift_report = th.compare("baseline.csv", "current.csv")
```

## Import Patterns

Truthound uses lazy loading for optimal import performance:

```python
# Core API - eagerly loaded (fast imports)
from truthound import check, scan, mask, profile, learn, Schema

# Advanced features - lazy loaded on first access
from truthound import compare      # Drift detection
from truthound import profiler     # Advanced profiling
from truthound import ml           # ML anomaly/drift detection
from truthound import lineage      # Data lineage tracking
from truthound import realtime     # Streaming validation
from truthound import checkpoint   # CI/CD integration
from truthound import datadocs     # HTML report generation

# Or access directly via module
import truthound as th
th.compare(...)           # Lazy loaded on first use
th.DataProfiler           # Lazy loaded on first use
```

## API Overview

### [Core Functions](core-functions.md)

Main entry points for data quality operations:

| Function | Description |
|----------|-------------|
| [`th.check()`](core-functions.md#thcheck) | Validate data quality |
| [`th.learn()`](core-functions.md#thlearn) | Learn schema from data |
| [`th.scan()`](core-functions.md#thscan) | Scan for PII |
| [`th.mask()`](core-functions.md#thmask) | Mask sensitive data |
| [`th.profile()`](core-functions.md#thprofile) | Generate data profile |
| [`th.compare()`](core-functions.md#thcompare) | Detect data drift |

### [Schema](schema.md)

Schema definition and validation:

| Class | Description |
|-------|-------------|
| `Schema` | Schema container with column definitions |
| `ColumnSchema` | Single column definition with constraints |

### [Validators](validators.md)

Validator interface and registration:

| Class | Description |
|-------|-------------|
| `Validator` | Base validator class |
| `ValidationIssue` | Issue representation |
| `Report` | Validation report container |

### [Data Sources](datasources.md)

Multi-backend data source support:

| Class | Description |
|-------|-------------|
| `BaseDataSource` | Base class for data sources |
| `PolarsDataSource` | Polars DataFrame source |
| `FileDataSource` | File-based source |
| `SQLiteDataSource` | SQLite database |
| `PostgreSQLDataSource` | PostgreSQL database |
| `BigQueryDataSource` | Google BigQuery |
| `SnowflakeDataSource` | Snowflake |

### [Reporters](reporters.md)

Output formatting:

| Class | Description |
|-------|-------------|
| `ConsoleReporter` | Terminal output |
| `JSONReporter` | JSON format |
| `HTMLReporter` | HTML reports |
| `JUnitXMLReporter` | CI/CD integration |

### [Advanced Features](advanced.md)

Enterprise features for ML, lineage, and streaming:

| Module | Description |
|--------|-------------|
| `truthound.ml` | ML anomaly/drift detection, rule learning |
| `truthound.lineage` | Data lineage tracking and visualization |
| `truthound.realtime` | Streaming and incremental validation |
| `truthound.profiler` | Advanced data profiling |
| `truthound.datadocs` | HTML report generation |
| `truthound.checkpoint` | CI/CD integration |

## Supported Input Types

The Python API accepts various input types:

```python
import truthound as th
import polars as pl
import pandas as pd

# File paths
report = th.check("data.csv")
report = th.check("data.parquet")
report = th.check("data.json")

# Polars DataFrame
df = pl.read_csv("data.csv")
report = th.check(df)

# Polars LazyFrame
lf = pl.scan_csv("data.csv")
report = th.check(lf)

# Pandas DataFrame
pdf = pd.read_csv("data.csv")
report = th.check(pdf)

# Dictionary
data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
report = th.check(data)

# DataSource (for databases)
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
)
report = th.check(source=source)
```

## Error Handling

```python
import truthound as th
from truthound.datasources.base import DataSourceError
from truthound.validators.base import (
    ValidationTimeoutError,
    ColumnNotFoundError,
    RegexValidationError,
)

try:
    report = th.check("data.csv")
    if report.issues:
        print(f"Found {len(report.issues)} issues")
except DataSourceError as e:
    print(f"Data source error: {e}")
except ValidationTimeoutError as e:
    print(f"Validation timed out: {e}")
except ColumnNotFoundError as e:
    print(f"Column not found: {e}")
```

## Type Hints

Truthound is fully typed. Use with mypy or pyright:

```python
# Core functions (eagerly loaded)
from truthound import check, learn, scan, mask, profile

# Drift comparison (lazy loaded)
from truthound import compare  # or: from truthound.drift import compare

# Types and classes
from truthound.schema import Schema, ColumnSchema
from truthound.validators.base import Validator, ValidationIssue
from truthound.report import Report
from truthound.datasources.base import BaseDataSource
from truthound.types import Severity

# Drift types
from truthound.drift.report import DriftReport, ColumnDrift
from truthound.drift.detectors import DriftResult, DriftLevel
```

## See Also

- [CLI Reference](../cli/index.md) - Command-line interface
- [Guides](../guides/index.md) - Usage guides
- [Tutorials](../tutorials/examples.md) - Step-by-step tutorials
