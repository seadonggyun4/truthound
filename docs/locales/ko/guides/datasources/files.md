# 파일-Based Data Sources

실무 운영 가이드에서 Truthound, JSON, CSV, Parquet, NDJSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Truthound, Polars, File-based, They을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Extension을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Scan, Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Eager, Loading을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-----------|---------------|---------------|
| 실무 운영 가이드에서 CSV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.csv`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pl.scan_csv()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.parquet`, `.pq`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pl.scan_parquet()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pl.read_json()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON, NDJSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.ndjson`, `.jsonl`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pl.scan_ndjson()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## FileDataSource

실무 운영 가이드에서 `FileDataSource`, FileDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### 설정

실무 운영 가이드에서 `FileDataSourceConfig`, FileDataSourceConfig을(를) 다루는 항목입니다:

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

## CSV 파일

실무 운영 가이드에서 CSV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

### 스키마 Inference

실무 운영 가이드에서 Polars을(를) 다루는 항목입니다:

```python
# Scan more rows for complex schemas
config = FileDataSourceConfig(infer_schema_length=50000)
source = FileDataSource("mixed_types.csv", config=config)

# Check inferred schema
print(source.schema)
```

## Parquet 파일

실무 운영 가이드에서 Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Parquet Loading

```python
from truthound.datasources import FileDataSource

source = FileDataSource("data.parquet")
# or
source = FileDataSource("data.pq")
```

### Parquet Features

실무 운영 가이드에서 Parquet을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 Parquet을(를) 다루는 항목입니다:

```python
config = FileDataSourceConfig(rechunk=True)
source = FileDataSource("data.parquet", config=config)
```

## JSON 파일

실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic JSON Loading

```python
from truthound.datasources import FileDataSource

source = FileDataSource("data.json")
```

### JSON 문서 구조

`.json` 파일은 하나의 완전한 JSON 문서여야 합니다.

```json
[
  {"id": 1, "name": "Alice", "age": 30},
  {"id": 2, "name": "Bob", "age": 25},
  {"id": 3, "name": "Charlie", "age": 35}
]
```

Truthound는 최상위 JSON 값을 다음 행 모델로 변환합니다.

| 최상위 값 | 행 모델 |
|-----------|---------|
| 객체 배열 | 객체 하나당 한 행 |
| 객체 | 객체 필드를 포함하는 한 행 |
| 스칼라 배열 | 배열 항목마다 `value` 한 행 |
| 스칼라 | `value` 열을 가진 한 행 |

중첩 객체와 배열은 Polars `Struct`, `List` 열로 보존됩니다. `.json`은 하나의
bounded 문서로 eager loading합니다. 대용량 또는 스트리밍 데이터는 line-delimited
lazy scan을 유지하는 `.ndjson` 또는 `.jsonl`을 사용하세요. 여러 JSON 문서를
`.json`에 연속해서 넣는 것은 허용하지 않으며 line-delimited 확장자를 사용해야
합니다.

## NDJSON / JSONL 파일

실무 운영 가이드에서 JSON, Newline-delimited, NDJSON/JSONL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic NDJSON Loading

```python
from truthound.datasources import FileDataSource

# Both extensions are supported
source = FileDataSource("data.ndjson")
source = FileDataSource("data.jsonl")
```

### NDJSON Structure

실무 운영 가이드에서 JSON을(를) 다루는 항목입니다:

```
{"id": 1, "name": "Alice", "age": 30}
{"id": 2, "name": "Bob", "age": 25}
{"id": 3, "name": "Charlie", "age": 35}
```

### NDJSON Advantages

- 실무 운영 가이드에서 `pl.scan_ndjson()`, Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Line-by-line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Schema을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
config = FileDataSourceConfig(
    infer_schema_length=10000,  # Scan first 10k lines
    ignore_errors=True,         # Skip malformed lines
    rechunk=True,               # Better memory layout
)
source = FileDataSource("large_data.ndjson", config=config)
```

## Working with LazyFrames

실무 운영 가이드에서 Polars, LazyFrame을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
source = FileDataSource("large_data.csv")

# Check if sampling is recommended
if source.needs_sampling():
    # Sample returns a PolarsDataSource (in-memory)
    sampled = source.sample(n=100_000, seed=42)

# Or use get_safe_sample()
safe_source = source.get_safe_sample()
```

> 실무 운영 가이드에서 Polars, `FileDataSource`, `PolarsDataSource`, Note, Sampling, FileDataSource, PolarsDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증 Example

실무 운영 가이드에서 API을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 Convenience을(를) 다루는 항목입니다:

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

| 실무 운영 가이드에서 Extension을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Notes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|--------|-------|
| 실무 운영 가이드에서 `.csv`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CSV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Comma-separated을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.parquet`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Columnar을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.pq`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Alternative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Array을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.ndjson`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, NDJSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, Newline-delimited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `.jsonl`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, JSONL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, Lines, NDJSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

> 실무 운영 가이드에서 `.xlsx`, `.xls`, `PandasDataSource`, Note, Excel, Convert, CSV, Parquet을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 권장 방식

1. 실무 운영 가이드에서 Parquet, Better을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 JSON, NDJSON, Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `infer_schema_length`, Adjust을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 `ignore_errors`, Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 `rechunk=True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
