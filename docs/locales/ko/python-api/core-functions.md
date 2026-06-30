# Core Functions

Python API 사용에서 Truthound, Main을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.read()

Python API 사용에서 Polars, `datasources.get_datasource()`, Reads, DataFrame, Convenience을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
def read(
    data: Any,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 File, DataFrame, DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Optional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `**kwargs`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `get_datasource()`, Additional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Supported Input Types

| Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|---------|
| Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일 path | Python API 사용에서 `th.read("data.csv")`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `pl.DataFrame`, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Polars, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `th.read(df)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `pl.LazyFrame`, LazyFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Polars, LazyFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `th.read(lf)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼-oriented data | Python API 사용에서 `th.read({"a": [1,2,3], "b": ["x","y","z"]})`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| `dict` (설정) | 설정 with "path" key | Python API 사용에서 `th.read({"path": "data.csv", "delimiter": ";"})`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `th.read(source)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

Python API 사용에서 Polars, `pl.DataFrame`, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 예시

```python
import truthound as th

# Read from file path
df = th.read("data.csv")
df = th.read("data.parquet")
df = th.read("data.json")

# Read from raw data dict
df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Read with config dict
df = th.read({"path": "data.csv", "delimiter": ";"})

# With sampling for large datasets
df = th.read("large_data.csv", sample_size=10000)

# Read from Polars DataFrame/LazyFrame (passthrough)
df = th.read(pl.DataFrame({"x": [1, 2, 3]}))

# With additional options
df = th.read("data.csv", has_header=False)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.check()

Python API 사용에서 Truthound, Validates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
def check(
    data: Any = None,
    source: BaseDataSource | None = None,
    context: TruthoundContext | None = None,
    validators: list[str | Validator] | None = None,
    validator_config: dict[str, dict[str, Any]] | None = None,
    min_severity: str | Severity | None = None,
    schema: str | Path | Schema | None = None,
    auto_schema: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
    result_format: str | ResultFormat | ResultFormatConfig = ResultFormat.SUMMARY,
    catch_exceptions: bool = True,
    max_retries: int = 0,
    exclude_columns: list[str] | None = None,
) -> ValidationRunResult:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Input, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `data`, DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `context`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 TruthoundContext, Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Explicit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `validators`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증기]` | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, Specific, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `validator_config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per-검증기 설정 dict |
| Python API 사용에서 `min_severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | `스키마 \ | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Path을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 to validate against |
| Python API 사용에서 `auto_schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Auto-learn and 캐시 스키마 |
| Python API 사용에서 `parallel`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 DAG-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `max_workers`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `pushdown`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 SQL, Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `result_format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ResultFormat을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ResultFormatConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `SUMMARY`, SUMMARY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Detail, VE-1을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `catch_exceptions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Isolate, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `max_retries`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Retry, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `exclude_columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `list[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Columns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

`ValidationRunResult` - Immutable 런타임 결과 with:

- Python API 사용에서 `checks`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `issues`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- `execution_issues`: 검증기 실패 captured by exception isolation
- Python API 사용에서 `execution_mode`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `planned_execution_mode`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `metadata`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 예시

```python
import truthound as th
from truthound.types import ResultFormat, ResultFormatConfig, Severity

# Basic validation through the auto-suite
run = th.check("data.csv")
print(f"Checks: {len(run.checks)}")
print(f"Issues: {len(run.issues)}")
print(f"Planned mode: {run.planned_execution_mode}")
print(f"Actual mode: {run.execution_mode}")

# With specific validators
run = th.check("data.csv", validators=["null", "duplicate", "range"])

# With validator configuration
run = th.check(
    "data.csv",
    validators=["regex"],
    validator_config={"regex": {"patterns": {"email": r"^[\w.+-]+@[\w-]+\.[\w.-]+$"}}}
)

# Exclude columns from all validators
run = th.check("users.csv", exclude_columns=["first_name", "last_name"])

# Per-validator column exclusion via validator_config
run = th.check(
    "users.csv",
    validator_config={"unique": {"exclude_columns": ["first_name"]}}
)

# With schema validation
schema = th.learn("baseline.csv")
run = th.check("data.csv", schema=schema)

# With DataSource
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
run = th.check(source=source)

# Filter by severity
run = th.check("data.csv", min_severity="medium")
critical_only = run.filter_by_severity(Severity.CRITICAL)

# Parallel execution for large datasets
run = th.check("data.csv", parallel=True, max_workers=4)

# Query pushdown for SQL sources
run = th.check(source=source, pushdown=True)

# Result format control (VE-1)
run = th.check("data.csv", result_format="boolean_only")  # Fastest, pass/fail only
run = th.check("data.csv", result_format=ResultFormat.COMPLETE)

# Fine-grained result format configuration
config = ResultFormatConfig(
    format=ResultFormat.COMPLETE,
    include_unexpected_rows=True,
    max_unexpected_rows=500,
    return_debug_query=True,
)
run = th.check("data.csv", result_format=config)

# Exception isolation with auto-retry (VE-5)
run = th.check("data.csv", catch_exceptions=True, max_retries=3)

# Work with structured issue and check results
for issue in run.issues:
    if issue.result:
        print(f"Elements: {issue.result.element_count}")
        print(f"Unexpected: {issue.result.unexpected_count} ({issue.result.unexpected_percent:.1%})")
    if issue.exception_info:
        print(f"Exception: {issue.exception_info.failure_category}")

# Reporter and docs helpers are available on the run result
# They are convenience facades that lazy-import outer reporter/datadocs services.
print(run.render(format="json"))
run.write("validation-run.json")
html = run.build_docs(title="Validation Overview")
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.learn()

Learns a 스키마 from data.

### Signature

```python
def learn(
    data: Any = None,
    source: BaseDataSource | None = None,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
) -> Schema:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Input, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `data`, DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `infer_constraints`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Infer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `categorical_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `20`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

Python API 사용에서 `Schema`, Schema, Inferred을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 예시

```python
import truthound as th

# Basic schema learning
schema = th.learn("baseline.csv")

# Without constraint inference (types only)
schema = th.learn("data.csv", infer_constraints=False)

# Custom categorical threshold
schema = th.learn("data.csv", categorical_threshold=50)

# Save schema to file
schema.save("schema.yaml")

# Use for validation
report = th.check("new_data.csv", schema=schema)

# From database using DataSource
from truthound.datasources.sql import SQLiteDataSource
source = SQLiteDataSource(database="mydb.db", table="users")
schema = th.learn(source=source)

# PostgreSQL example
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
schema = th.learn(source=source)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.scan()

Python API 사용에서 Scans, PII을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
def scan(
    data: Any = None,
    source: BaseDataSource | None = None,
) -> PIIReport:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Input, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

`PIIReport` - 리포트 with PII findings.

### 예시

```python
import truthound as th

# Basic PII scan
pii_report = th.scan("customers.csv")

# findings is a list of dicts
for finding in pii_report.findings:
    print(f"{finding['column']}: {finding['pii_type']} ({finding['confidence']}%)")

# Check if PII was detected
if pii_report.has_pii:
    print(f"Found {len(pii_report.findings)} columns with potential PII")

# Print formatted report
pii_report.print()

# From database
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
pii_report = th.scan(source=source)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.mask()

Python API 사용에서 Masks, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
def mask(
    data: Any = None,
    source: BaseDataSource | None = None,
    columns: list[str] | None = None,
    strategy: str = "redact",
    *,
    strict: bool = False,
) -> pl.DataFrame:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Input, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `list[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 to mask (auto-detect if None) |
| Python API 사용에서 `strategy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `"redact"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Masking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `strict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Raise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Strategies

| Python API 사용에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|---------|
| Python API 사용에서 `redact`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Replace을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `john@example.com`, `****`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `hash`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `john@example.com`, `a8b9c0d1e2f3g4h5`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `fake`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Hash-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `john@example.com`, `user_a8b9c0d1@fake.com`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

Python API 사용에서 `pl.DataFrame`, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 예시

```python
import truthound as th
import polars as pl

df = pl.read_csv("customers.csv")

# Auto-detect and mask PII
masked = th.mask(df)

# Specific columns
masked = th.mask(df, columns=["email", "phone", "ssn"])

# Hash strategy (deterministic)
masked = th.mask(df, strategy="hash")

# Fake data (for testing)
masked = th.mask(df, strategy="fake")

# Strict mode - fails if column doesn't exist
masked = th.mask(df, columns=["email"], strict=True)

# From file path
masked = th.mask("customers.csv", columns=["email"])
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## th.프로파일()

Python API 사용에서 Generates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
def profile(
    data: Any = None,
    source: BaseDataSource | None = None,
) -> ProfileReport:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Input, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `BaseDataSource`, BaseDataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Returns

`ProfileReport` - 프로파일 with 컬럼 statistics.

### 예시

```python
import truthound as th

profile = th.profile("data.csv")

print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
print(f"Size: {profile.size_bytes} bytes")

# columns is a list of dicts with column statistics
for col in profile.columns:
    print(f"{col['name']}:")
    print(f"  Type: {col['dtype']}")
    print(f"  Nulls: {col['null_pct']}")
    print(f"  Unique: {col['unique_pct']}")
    if 'min' in col:
        print(f"  Min: {col['min']}")
        print(f"  Max: {col['max']}")

# Print formatted report
profile.print()

# Export to JSON
json_output = profile.to_json(indent=2)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## compare() in truthound.드리프트

Python API 사용에서 `truthound.drift`, `truthound`, Drift을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Signature

```python
from truthound.drift import compare

def compare(
    baseline: Any,
    current: Any,
    columns: list[str] | None = None,
    method: str = "auto",
    threshold: float | None = None,
    sample_size: int | None = None,
) -> DriftReport:
```

### Parameters

| Python API 사용에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| Python API 사용에서 `baseline`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Baseline을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `current`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Any`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Current을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `list[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Columns, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `method`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `"auto"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Detection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Custom 드리프트 threshold |
| Python API 사용에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Methods

Python API 사용에서 Statistical, Tests을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 Type | Python API 사용에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|-------------|----------|
| Python API 사용에서 `auto`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `ks`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Kolmogorov-Smirnov을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Continuous을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `psi`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Population, Stability, Index을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | ML 모니터링 |
| Python API 사용에서 `chi2`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Chi-squared을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Categorical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Categorical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `cvm`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Mises을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Tail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `anderson`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Anderson-Darling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Extreme을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

**Divergence 메트릭:**

| Python API 사용에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 Type | Python API 사용에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|-------------|----------|
| Python API 사용에서 `js`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Jensen-Shannon을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `kl`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Kullback-Leibler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Information을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

**Distance 메트릭:**

| Python API 사용에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 Type | Python API 사용에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|-------------|----------|
| Python API 사용에서 `wasserstein`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Wasserstein, Earth, Mover을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Intuitive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `hellinger`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Hellinger을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Bounded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `bhattacharyya`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Bhattacharyya을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Classification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `tv`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Total, Variation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `energy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Energy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Location/scale을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `mmd`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Maximum, Mean, Discrepancy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 High-dimensional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

> Python API 사용에서 `ks`, `psi`, `kl`, `wasserstein`, `cvm`, `anderson`, `energy`, `mmd`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> For non-numeric 컬럼, use `method="auto"`, `method="chi2"`, `method="js"`, `method="hellinger"`, `method="bhattacharyya"`, or `method="tv"`.

### Returns

`DriftReport` - Per-컬럼 드리프트 analysis.

### 예시

```python
from truthound.drift import compare

# Basic comparison
drift = compare("baseline.csv", "current.csv")

if drift.has_high_drift:
    print("Significant drift detected!")
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"  {col_drift.column}: {col_drift.result.method} = {col_drift.result.statistic:.4f}")

# Specific method (psi requires numeric columns)
drift = compare("train.csv", "prod.csv", method="psi", columns=["age", "income", "score"])

# KL divergence
drift = compare("baseline.csv", "current.csv", method="kl", columns=["age", "income"])

# Wasserstein distance (normalized)
drift = compare("baseline.csv", "current.csv", method="wasserstein", columns=["age", "income"])

# Cramér-von Mises (sensitive to tails)
drift = compare("baseline.csv", "current.csv", method="cvm", columns=["age", "income"])

# Anderson-Darling (most sensitive to tail differences)
drift = compare("baseline.csv", "current.csv", method="anderson", columns=["age", "income"])

# With custom threshold
drift = compare("old.csv", "new.csv", threshold=0.1)

# For large datasets, use sampling
drift = compare("big_train.csv", "big_prod.csv", sample_size=10000)

# Compare specific columns
drift = compare("baseline.csv", "current.csv", columns=["age", "income", "score"])

# Get drifted column names
drifted_cols = drift.get_drifted_columns()
print(f"Drifted columns: {drifted_cols}")
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## DriftReport

리포트 returned by `compare()` from `truthound.drift`.

### Definition

```python
from truthound.drift.report import DriftReport, ColumnDrift
from truthound.drift.detectors import DriftResult, DriftLevel

@dataclass
class DriftReport:
    """Complete drift detection report."""

    baseline_source: str
    current_source: str
    baseline_rows: int
    current_rows: int
    columns: list[ColumnDrift]
```

### Properties

| Python API 사용에서 Property을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|------|-------------|
| Python API 사용에서 `has_drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | True if any 컬럼 has 드리프트 |
| Python API 사용에서 `has_high_drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 True, HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Methods

| Python API 사용에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Returns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|---------|-------------|
| Python API 사용에서 `get_drifted_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `list[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 names with detected 드리프트 |
| Python API 사용에서 `to_dict()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `to_json(indent)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 JSON, Convert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `print()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Print formatted 리포트 to stdout |

### ColumnDrift

```python
@dataclass
class ColumnDrift:
    """Drift information for a single column."""

    column: str              # Column name
    dtype: str               # Data type
    result: DriftResult      # Drift detection result
    baseline_stats: dict     # Baseline statistics
    current_stats: dict      # Current statistics
```

### DriftResult

```python
@dataclass
class DriftResult:
    """Result of a drift detection test."""

    statistic: float         # Test statistic value
    p_value: float | None    # P-value (if applicable)
    threshold: float         # Threshold used for detection
    drifted: bool            # True if drift detected
    level: DriftLevel        # Drift severity level
    method: str              # Detection method used
    details: str | None      # Additional details
```

### DriftLevel Enum

```python
from truthound.drift.detectors import DriftLevel

class DriftLevel(str, Enum):
    NONE = "none"      # No drift
    LOW = "low"        # Minor drift
    MEDIUM = "medium"  # Moderate drift
    HIGH = "high"      # Significant drift
```

### 사용 예시

```python
from truthound.drift import compare

drift = compare("baseline.csv", "current.csv")

# Check overall drift
print(f"Has drift: {drift.has_drift}")
print(f"Has high drift: {drift.has_high_drift}")

# Iterate columns
for col in drift.columns:
    r = col.result
    print(f"{col.column} ({col.dtype}):")
    print(f"  Method: {r.method}")
    print(f"  Statistic: {r.statistic:.4f}")
    print(f"  Level: {r.level.value}")
    print(f"  Drifted: {r.drifted}")

# Export
json_output = drift.to_json(indent=2)
dict_output = drift.to_dict()
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ResultFormat (VE-1)

Python API 사용에서 Great Expectations, `result_format`, Controls, Great, Expectations을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### ResultFormat Enum

```python
from truthound.types import ResultFormat

class ResultFormat(str, Enum):
    BOOLEAN_ONLY = "boolean_only"  # Only pass/fail flag
    BASIC = "basic"                # + observed_value, unexpected_count, samples
    SUMMARY = "summary"            # + value_counts, index_list (DEFAULT)
    COMPLETE = "complete"          # + full unexpected_list, unexpected_rows, debug_query
```

Python API 사용에서 Enrichment, Phases을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Data, Collected을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------|----------------|
| Python API 사용에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Always을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `element_count`, `missing_count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 BASIC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `partial_unexpected_list`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 SUMMARY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `partial_unexpected_counts`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 COMPLETE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `unexpected_rows`, `debug_query`, DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### ResultFormatConfig

```python
from truthound.types import ResultFormatConfig

config = ResultFormatConfig(
    format=ResultFormat.SUMMARY,          # Detail level
    partial_unexpected_count=20,          # Max samples to collect
    include_unexpected_rows=False,        # Include full DataFrame of failing rows
    max_unexpected_rows=1000,             # Max rows in unexpected_rows
    include_unexpected_index=False,       # Include row indices
    return_debug_query=False,             # Include Polars query string
)

# Factory from any input type
config = ResultFormatConfig.from_any("complete")
config = ResultFormatConfig.from_any(ResultFormat.BASIC)
config = ResultFormatConfig.from_any(None)  # Returns default SUMMARY
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Type Definitions

### DataInput

```python
from truthound.types import DataInput

DataInput = Union[
    str,           # File path
    pl.DataFrame,  # Polars DataFrame
    pl.LazyFrame,  # Polars LazyFrame
    pd.DataFrame,  # pandas DataFrame (via Any)
    dict,          # Dictionary
]
```

### Severity

```python
from truthound.types import Severity

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Severity supports comparison operators (custom __ge__, __gt__, __le__, __lt__)
Severity.HIGH > Severity.LOW      # True
Severity.MEDIUM >= Severity.LOW   # True
Severity.CRITICAL > Severity.HIGH # True
Severity.LOW < Severity.MEDIUM    # True
```

!!! info "참고"
Python API 사용에서 Truthound, `INFO`, `LOW`, INFO, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [스키마](schema.md) - 스키마 classes
- [검증기](validators.md) - 검증기 interface
- Python API 사용에서 Data, Sources, Database을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 CLI, Reference, Command-line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
