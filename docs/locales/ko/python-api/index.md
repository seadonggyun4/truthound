# Python API 레퍼런스

Python API 사용에서 Truthound, API, Python, Core을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 `truthound`, Host-native을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 Truthound, API, APIs, Orchestration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 Truthound, API, Repository-console, APIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Python API 사용 개요

Python API 사용에서 Choose, Python을(를) 다루는 항목입니다:

- Python API 사용에서 SQL, SQL-backed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Truthound, Core을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.drift`, `truthound.checkpoint`, `truthound.reporters`, `truthound.profiler`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설치

```bash
pip install truthound
```

Python API 사용에서 Optional을(를) 다루는 항목입니다:

```bash
pip install truthound[ai]
```

## 빠른 시작

```python
import truthound as th
from truthound.drift import compare

# Validate data through the 3.0 kernel
run = th.check("data.csv")

# Learn a reusable baseline schema
schema = th.learn("baseline.csv")

# Scan for PII
pii_report = th.scan("customers.csv")

# Mask sensitive data
masked_df = th.mask(run.source, strategy="hash")

# Profile data
profile = th.profile("data.csv")

# Compare datasets for drift
drift = compare("baseline.csv", "current.csv")
```

## Core 3.0 Mental Model

Python API 사용에서 Truthound을(를) 다루는 항목입니다:

- Python API 사용에서 `truthound`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 ValidationRunResult, `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.drift`, `truthound.checkpoint`, `truthound.reporters`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `truthound.ai`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 `render()`, `write()`, `build_docs()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 Core, Functions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
런타임 contract first.

## Import Patterns

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound import (
    check,
    scan,
    mask,
    profile,
    read,
    learn,
    Schema,
    TruthoundContext,
    ValidationRunResult,
)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound.drift import compare
from truthound.reporters import get_reporter
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.profiler import profile_data
import truthound.ai as thai
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
import truthound as th
import truthound.checkpoint as checkpoint
import truthound.drift as drift
```

## Recommended Reading Path

1. Python API 사용에서 Core, Functions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. [스키마](schema.md)
3. [검증기](validators.md)
4. [데이터 소스](datasources.md)
5. [리포터](reporters.md)
6. Python API 사용에서 Advanced, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 Truthound, Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 개요

### Root Facade

| Python API 사용에서 Symbol을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| Python API 사용에서 `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ValidationRunResult, Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `th.learn()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `th.scan()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Scan, PII을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `th.mask()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Mask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `th.profile()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Generate a data 프로파일 |
| Python API 사용에서 `th.read()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Polars, Load을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Canonical 검증 런타임 output |
| Python API 사용에서 TruthoundContext, Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Zero-config을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Core-Adjacent Namespaces

| Python API 사용에서 Namespace을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| Python API 사용에서 `truthound.drift`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 드리프트 comparison via `compare()` and `DriftReport` |
| Python API 사용에서 `truthound.checkpoint`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 오케스트레이션, actions, and CI 통합 |
| Python API 사용에서 `truthound.reporters`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Rendering을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `truthound.profiler`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Profiling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `truthound.datadocs`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 HTML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `truthound.ai`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Optional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `truthound.lineage`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Lineage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `truthound.realtime`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Streaming and incremental 검증 |
| Python API 사용에서 `truthound.ml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | ML-assisted 이상치 and 드리프트 tooling |

### Core Types

| Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| Python API 사용에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Immutable 검증 run 결과 |
| Python API 사용에서 `CheckResult`, CheckResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Per-check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `ValidationIssue`, ValidationIssue을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Individual issue emitted by a 검증기 |
| Python API 사용에서 `Schema`, Schema을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Schema을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Supported Input Types

Python API 사용에서 API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
import pandas as pd
import polars as pl
import truthound as th

# File paths
run = th.check("data.csv")
run = th.check("data.parquet")
run = th.check("data.json")

# Polars DataFrame / LazyFrame
df = pl.read_csv("data.csv")
run = th.check(df)
run = th.check(df.lazy())

# Pandas DataFrame
pdf = pd.read_csv("data.csv")
run = th.check(pdf)

# Dictionary input
run = th.check({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

# SQL-backed DataSource
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
)
run = th.check(source=source)
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
    run = th.check("data.csv", catch_exceptions=False)
    if run.issues:
        print(f"Found {len(run.issues)} issues")
except DataSourceError as exc:
    print(f"Data source error: {exc}")
except ValidationTimeoutError as exc:
    print(f"Validation timed out: {exc}")
except ColumnNotFoundError as exc:
    print(f"Column not found: {exc}")
except RegexValidationError as exc:
    print(f"Invalid regex: {exc}")
```

## Type Hints

Python API 사용에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound import ValidationRunResult, check, learn, mask, profile, read, scan
from truthound.core.results import CheckResult
from truthound.datasources.base import BaseDataSource
from truthound.drift import ColumnDrift, DriftReport, compare
from truthound.schema import Schema, ColumnSchema
from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
```

## 함께 보기

- Python API 사용에서 CLI, API, Reference, Overview, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 CLI, Reference, Command-line을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Guides, Task-oriented을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Tutorials, Step-by-step을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Migration, Removed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
