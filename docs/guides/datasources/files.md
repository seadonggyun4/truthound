# File-Based Data Sources

This document covers file-based data sources in Truthound, including CSV, Parquet, JSON, and NDJSON formats.

## Overview

File-based data sources are the simplest way to load data into Truthound. They use Polars' lazy evaluation for efficient processing of large files.

| Format | Extension | Scan Function | Eager Loading |
|--------|-----------|---------------|---------------|
| CSV | `.csv` | `pl.scan_csv()` | No |
| Parquet | `.parquet`, `.pq` | `pl.scan_parquet()` | No |
| JSON | `.json` | `pl.read_json()` | Yes |
| NDJSON | `.ndjson`, `.jsonl` | `pl.scan_ndjson()` | No |

## FileDataSource

The `FileDataSource` class handles all file-based data loading.

### Basic Usage

```python
from truthound.datasources import FileDataSource

# CSV file
source = FileDataSource("data.csv")

# Parquet file
source = FileDataSource("data.parquet")

# JSON file
source = FileDataSource("data.json")

# NDJSON / JSONL file
source = FileDataSource("data.ndjson")
source = FileDataSource("data.jsonl")
```

### Properties

```python
source = FileDataSource("users.csv")

# Get file path
print(source.path)        # PosixPath('users.csv')

# Get detected file type
print(source.file_type)   # 'csv'

# Get data source name (defaults to filename)
print(source.name)        # 'users.csv'

# Get schema
print(source.schema)
# {'id': ColumnType.INTEGER, 'name': ColumnType.STRING, ...}

# Get row count
print(source.row_count)   # 10000

# Get columns
print(source.columns)     # ['id', 'name', 'email', ...]
```

### Configuration

`FileDataSourceConfig` provides options for file parsing:

```python
from truthound.datasources import FileDataSource, FileDataSourceConfig

config = FileDataSourceConfig(
    # Schema inference
    infer_schema_length=10000,  # Rows to scan for schema (default: 10000)
    ignore_errors=False,        # Skip malformed rows (default: False)

    # CSV-specific options
    encoding="utf8",            # File encoding (default: "utf8")
    separator=",",              # Column separator (default: ",")

    # Performance options
    rechunk=False,              # Rechunk for better memory layout
    streaming=False,            # Use streaming mode for large files

    # Size limits
    max_rows=10_000_000,        # Maximum rows allowed
    max_memory_mb=4096,         # Maximum memory in MB
    sample_size=100_000,        # Default sample size
    sample_seed=42,             # Reproducible sampling
)

source = FileDataSource("large_data.csv", config=config)
```

## CSV Files

CSV is the most common format with configurable parsing options.

### Basic CSV Loading

```python
from truthound.datasources import FileDataSource

source = FileDataSource("data.csv")
engine = source.get_execution_engine()
print(f"Rows: {engine.count_rows()}")
```

### Custom Delimiters

```python
from truthound.datasources import FileDataSource, FileDataSourceConfig

# Tab-separated values
config = FileDataSourceConfig(separator="\t")
source = FileDataSource("data.tsv", config=config)

# Pipe-separated values
config = FileDataSourceConfig(separator="|")
source = FileDataSource("data.psv", config=config)

# Semicolon-separated (European locale)
config = FileDataSourceConfig(separator=";")
source = FileDataSource("data.csv", config=config)
```

### Encoding Options

```python
# UTF-8 (default)
config = FileDataSourceConfig(encoding="utf8")

# Latin-1
config = FileDataSourceConfig(encoding="iso-8859-1")

# Windows encoding
config = FileDataSourceConfig(encoding="cp1252")

source = FileDataSource("legacy_data.csv", config=config)
```

### Handling Malformed Data

```python
# Skip malformed rows instead of failing
config = FileDataSourceConfig(ignore_errors=True)
source = FileDataSource("messy_data.csv", config=config)
```

### Schema Inference

Polars infers schema by scanning initial rows:

```python
# Scan more rows for complex schemas
config = FileDataSourceConfig(infer_schema_length=50000)
source = FileDataSource("mixed_types.csv", config=config)

# Check inferred schema
print(source.schema)
```

## Parquet Files

Parquet is a columnar format optimized for analytics workloads.

### Basic Parquet Loading

```python
from truthound.datasources import FileDataSource

source = FileDataSource("data.parquet")
# or
source = FileDataSource("data.pq")
```

### Parquet Features

Parquet files support efficient metadata queries:

```python
source = FileDataSource("data.parquet")

# Row count is available from metadata (no scan needed)
print(source.row_count)

# Schema is available from metadata
print(source.schema)

# Capabilities include ROW_COUNT
from truthound.datasources import DataSourceCapability
print(DataSourceCapability.ROW_COUNT in source.capabilities)  # True
```

### Rechunking

For better memory layout with multiple Parquet files:

```python
config = FileDataSourceConfig(rechunk=True)
source = FileDataSource("data.parquet", config=config)
```

## JSON Files

JSON files are loaded eagerly (entire file into memory).

### Basic JSON Loading

```python
from truthound.datasources import FileDataSource

source = FileDataSource("data.json")
```

### JSON Structure

The JSON file should be an array of objects:

```json
[
  {"id": 1, "name": "Alice", "age": 30},
  {"id": 2, "name": "Bob", "age": 25},
  {"id": 3, "name": "Charlie", "age": 35}
]
```

> **Note**: JSON files are loaded eagerly using `pl.read_json()` since Polars doesn't have a `scan_json()` function. For large JSON datasets, consider using NDJSON format.

## NDJSON / JSONL Files

Newline-delimited JSON (NDJSON/JSONL) is ideal for large datasets.

### Basic NDJSON Loading

```python
from truthound.datasources import FileDataSource

# Both extensions are supported
source = FileDataSource("data.ndjson")
source = FileDataSource("data.jsonl")
```

### NDJSON Structure

Each line is a separate JSON object:

```
{"id": 1, "name": "Alice", "age": 30}
{"id": 2, "name": "Bob", "age": 25}
{"id": 3, "name": "Charlie", "age": 35}
```

### NDJSON Advantages

- Supports lazy loading (`pl.scan_ndjson()`)
- Line-by-line processing for large files
- Schema inference from sample rows

```python
config = FileDataSourceConfig(
    infer_schema_length=10000,  # Scan first 10k lines
    ignore_errors=True,         # Skip malformed lines
    rechunk=True,               # Better memory layout
)
source = FileDataSource("large_data.ndjson", config=config)
```

## Working with LazyFrames

All file sources provide access to the underlying Polars LazyFrame:

```python
source = FileDataSource("data.csv")

# Get LazyFrame for custom operations
lf = source.to_polars_lazyframe()

# Chain Polars operations
result = (
    lf
    .filter(pl.col("age") > 25)
    .select(["id", "name", "age"])
    .collect()
)
```

## Sampling

For large files, use sampling to reduce memory usage:

```python
source = FileDataSource("large_data.csv")

# Check if sampling is recommended
if source.needs_sampling():
    # Sample returns a PolarsDataSource (in-memory)
    sampled = source.sample(n=100_000, seed=42)

# Or use get_safe_sample()
safe_source = source.get_safe_sample()
```

> **Note**: Sampling from a `FileDataSource` returns a `PolarsDataSource` since the sampled data is loaded into memory.

## Validation Example

Using file sources with the validation API:

```python
import truthound as th
from truthound.datasources import FileDataSource

# Create file source
source = FileDataSource("users.csv")

# Run validation
report = th.check(
    source=source,
    validators=["null", "duplicate"],
    columns=["id", "email"],
)

# Or with rules
report = th.check(
    source=source,
    rules={
        "id": ["not_null", "unique"],
        "email": ["not_null", {"type": "regex", "pattern": r".*@.*"}],
        "age": [{"type": "range", "min": 0, "max": 150}],
    },
)

print(f"Found {len(report.issues)} issues")
```

## Factory Functions

Convenience functions for creating file sources:

```python
from truthound.datasources import from_file, get_datasource

# Using from_file
source = from_file("data.csv")
source = from_file("data.parquet")

# Using get_datasource (auto-detects file type)
source = get_datasource("data.csv")
source = get_datasource("data.json")
```

## Error Handling

```python
from truthound.datasources import FileDataSource
from truthound.datasources.base import DataSourceError

try:
    source = FileDataSource("missing.csv")
except DataSourceError as e:
    print(f"Error: {e}")  # "File not found: missing.csv"

try:
    source = FileDataSource("data.xlsx")  # Unsupported
except DataSourceError as e:
    print(f"Error: {e}")
    # "Unsupported file type: .xlsx. Supported: ['.csv', '.json', '.parquet', '.pq', '.ndjson', '.jsonl']"
```

## Supported Extensions

| Extension | Format | Notes |
|-----------|--------|-------|
| `.csv` | CSV | Comma-separated values |
| `.parquet` | Parquet | Columnar binary format |
| `.pq` | Parquet | Alternative extension |
| `.json` | JSON | Array of objects |
| `.ndjson` | NDJSON | Newline-delimited JSON |
| `.jsonl` | JSONL | JSON Lines (same as NDJSON) |

> **Note**: Excel files (`.xlsx`, `.xls`) are not directly supported. Convert to CSV or Parquet first, or use Pandas to load and then convert to a `PandasDataSource`.

## Best Practices

1. **Use Parquet for large datasets** - Better compression and faster reads
2. **Use NDJSON over JSON for streaming** - Supports lazy loading
3. **Adjust `infer_schema_length`** for complex schemas
4. **Enable `ignore_errors`** for messy real-world data
5. **Use `rechunk=True`** when combining multiple files
6. **Sample before validation** for exploratory analysis on large files
