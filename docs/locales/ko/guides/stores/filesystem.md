# FileSystem & Memory 스토어

실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## FileSystem Store

실무 운영 가이드에서 JSON, `FileSystemStore`, FileSystemStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
import truthound as th
from truthound.stores.backends.filesystem import FileSystemStore
from truthound.stores.results import ValidationResult

# Default configuration
store = FileSystemStore()

# Custom path
store = FileSystemStore(
    base_path=".truthound/store",
    namespace="production",
    prefix="validations",
)

# Save a result
run = th.check("customers.csv")
stored_result = ValidationResult.from_report(run, "customers.csv")
run_id = store.save(stored_result)

# Retrieve
result = store.get(run_id)
```

### Using the Factory

```python
from truthound.stores import get_store

store = get_store("filesystem", base_path=".truthound/store")
```

### 설정

```python
from truthound.stores.backends.filesystem import FileSystemConfig

config = FileSystemConfig(
    base_path=".truthound/store",    # Base directory
    namespace="default",              # Namespace for organization
    prefix="validations",             # Path prefix
    file_extension=".json",           # File extension
    create_dirs=True,                 # Auto-create directories
    pretty_print=True,                # Indent JSON output
    use_compression=False,            # Enable gzip compression
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `base_path`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.truthound/store`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Base directory for 파일 |
| 실무 운영 가이드에서 `namespace`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"default"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Logical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `prefix`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"validations"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Additional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `file_extension`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `.json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일 extension |
| 실무 운영 가이드에서 `create_dirs`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Create을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `pretty_print`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `use_compression`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Compress 파일 with gzip |

### Compression

실무 운영 가이드에서 Enable을(를) 다루는 항목입니다:

```python
store = FileSystemStore(
    base_path=".truthound/store",
    compression=True,  # Files saved as .json.gz
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:
- 실무 운영 가이드에서 `.json.gz`, Files을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `gzip.compress()`, Content을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Decompression을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Directory Structure

The store organizes 파일 as:

```
{base_path}/
├── {namespace}/
│   └── {prefix}/
│       ├── _index.json          # Metadata index
│       ├── {run_id}.json        # Result files
│       └── {run_id}.json.gz     # (if compression enabled)
```

### Index Management

실무 운영 가이드에서 `_index.json`을(를) 다루는 항목입니다:

```json
{
  "run-123": {
    "data_asset": "customers.csv",
    "run_time": "2024-01-15T10:30:00",
    "status": "failure",
    "file": "run-123.json",
    "tags": {"env": "prod"}
  }
}
```

실무 운영 가이드에서 Rebuild을(를) 다루는 항목입니다:

```python
store = FileSystemStore(base_path=".truthound/store")
store.initialize()
count = store.rebuild_index()
print(f"Indexed {count} items")
```

### Expectation Store

실무 운영 가이드에서 Store을(를) 다루는 항목입니다:

```python
from truthound.stores.backends.filesystem import FileSystemExpectationStore

store = FileSystemExpectationStore(
    base_path=".truthound/expectations",
    namespace="default",
    prefix="suites",
)

# Save suite
suite = ExpectationSuite.create("my_suite", "customers.csv")
store.save(suite)

# Retrieve
suite = store.get("my_suite")
```

## Memory Store

실무 운영 가이드에서 `MemoryStore`, MemoryStore, Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Use Cases

- 실무 운영 가이드에서 Unit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Development을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Temporary 검증 workflows
- 성능 benchmarking

### Basic Usage

```python
from truthound.stores.backends.memory import MemoryStore

store = MemoryStore()

# Save and retrieve
run_id = store.save(result)
result = store.get(run_id)

# Check existence
assert store.exists(run_id)

# Clear all data
count = store.clear_all()
```

### 설정

```python
from truthound.stores.backends.memory import MemoryConfig

config = MemoryConfig(
    max_items=0,       # 0 = unlimited
    deep_copy=True,    # Deep copy on save/retrieve
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `max_items`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `deep_copy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Deep을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Memory Management

실무 운영 가이드에서 `max_items`, Limit을(를) 다루는 항목입니다:

```python
store = MemoryStore(max_items=1000)

# When limit is reached, oldest items are removed
for i in range(1500):
    store.save(result)

# Only the newest 1000 items remain
assert len(store.list_ids()) == 1000
```

### Deep Copy Behavior

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
# With deep_copy=True (default)
store = MemoryStore(deep_copy=True)
run_id = store.save(result)
retrieved = store.get(run_id)
retrieved.tags["modified"] = True
original = store.get(run_id)
assert "modified" not in original.tags  # Original unchanged

# With deep_copy=False (faster but mutable)
store = MemoryStore(deep_copy=False)
# Warning: Changes to retrieved items affect stored data
```

### Expectation Store

```python
from truthound.stores.backends.memory import MemoryExpectationStore

store = MemoryExpectationStore()

suite = ExpectationSuite.create("test_suite", "data.csv")
store.save(suite)
suite = store.get("test_suite")
```

## Common Operations

실무 운영 가이드에서 `ValidationStore`, Both, ValidationStore을(를) 다루는 항목입니다:

```python
# Initialize (lazy, called automatically)
store.initialize()

# Save
run_id = store.save(result)

# Retrieve
result = store.get(run_id)

# Check existence
exists = store.exists(run_id)

# Delete
deleted = store.delete(run_id)

# List IDs
ids = store.list_ids()

# Query with filters
from truthound.stores.base import StoreQuery

query = StoreQuery(
    data_asset="customers.csv",
    status="failure",
    limit=10,
)
results = store.query(query)
```

## Error Handling

```python
from truthound.stores.base import (
    StoreNotFoundError,
    StoreReadError,
    StoreWriteError,
)

try:
    result = store.get("nonexistent-id")
except StoreNotFoundError as e:
    print(f"Not found: {e.identifier}")

try:
    store.save(result)
except StoreWriteError as e:
    print(f"Write failed: {e}")
```

## Choosing Between 스토어

| 실무 운영 가이드에서 Scenario을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Recommended, Store을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------------|
| 실무 운영 가이드에서 Production을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `FileSystemStore`, FileSystemStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Development을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `FileSystemStore`, FileSystemStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Unit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `MemoryStore`, MemoryStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 통합 tests | 실무 운영 가이드에서 `MemoryStore`, `FileSystemStore`, MemoryStore, FileSystemStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CI/CD 파이프라인 | 실무 운영 가이드에서 `FileSystemStore`, FileSystemStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Temporary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `MemoryStore`, MemoryStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 다음 단계

- 실무 운영 가이드에서 Cloud, Storage, GCS, Azure, Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Versioning, Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Caching, In-memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
