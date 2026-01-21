# FileSystem & Memory Stores

Local storage backends for development, testing, and single-node deployments.

## FileSystem Store

The `FileSystemStore` persists validation results as JSON files on the local filesystem. It requires no external dependencies and is the default backend.

### Basic Usage

```python
from truthound.stores.backends.filesystem import FileSystemStore

# Default configuration
store = FileSystemStore()

# Custom path
store = FileSystemStore(
    base_path=".truthound/store",
    namespace="production",
    prefix="validations",
)

# Save a result
result = ValidationResult.from_report(report, "customers.csv")
run_id = store.save(result)

# Retrieve
result = store.get(run_id)
```

### Using the Factory

```python
from truthound.stores import get_store

store = get_store("filesystem", base_path=".truthound/store")
```

### Configuration

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

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_path` | `str` | `.truthound/store` | Base directory for files |
| `namespace` | `str` | `"default"` | Logical grouping namespace |
| `prefix` | `str` | `"validations"` | Additional path prefix |
| `file_extension` | `str` | `.json` | File extension |
| `create_dirs` | `bool` | `True` | Create directories if missing |
| `pretty_print` | `bool` | `True` | Format JSON with indentation |
| `use_compression` | `bool` | `False` | Compress files with gzip |

### Compression

Enable gzip compression to reduce storage space:

```python
store = FileSystemStore(
    base_path=".truthound/store",
    compression=True,  # Files saved as .json.gz
)
```

When compression is enabled:
- Files are saved with `.json.gz` extension
- Content is compressed using `gzip.compress()`
- Decompression is automatic on read

### Directory Structure

The store organizes files as:

```
{base_path}/
├── {namespace}/
│   └── {prefix}/
│       ├── _index.json          # Metadata index
│       ├── {run_id}.json        # Result files
│       └── {run_id}.json.gz     # (if compression enabled)
```

### Index Management

The store maintains an `_index.json` file for fast queries:

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

Rebuild the index if it becomes corrupted:

```python
store = FileSystemStore(base_path=".truthound/store")
store.initialize()
count = store.rebuild_index()
print(f"Indexed {count} items")
```

### Expectation Store

Store expectation suites separately:

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

The `MemoryStore` keeps data in memory. Data is not persisted between sessions.

### Use Cases

- Unit testing
- Development and prototyping
- Temporary validation workflows
- Performance benchmarking

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

### Configuration

```python
from truthound.stores.backends.memory import MemoryConfig

config = MemoryConfig(
    max_items=0,       # 0 = unlimited
    deep_copy=True,    # Deep copy on save/retrieve
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_items` | `int` | `0` | Max items to store (0 = unlimited) |
| `deep_copy` | `bool` | `True` | Deep copy items on save/retrieve |

### Memory Management

Limit memory usage with `max_items`:

```python
store = MemoryStore(max_items=1000)

# When limit is reached, oldest items are removed
for i in range(1500):
    store.save(result)

# Only the newest 1000 items remain
assert len(store.list_ids()) == 1000
```

### Deep Copy Behavior

By default, items are deep copied to prevent mutation:

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

Both stores implement the `ValidationStore` protocol:

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

## Choosing Between Stores

| Scenario | Recommended Store |
|----------|-------------------|
| Production (single node) | `FileSystemStore` |
| Development | `FileSystemStore` |
| Unit tests | `MemoryStore` |
| Integration tests | `MemoryStore` or `FileSystemStore` |
| CI/CD pipelines | `FileSystemStore` |
| Temporary workflows | `MemoryStore` |

## Next Steps

- [Cloud Storage](cloud-storage.md) - S3, GCS, Azure Blob for distributed systems
- [Versioning](versioning.md) - Version history for results
- [Caching](caching.md) - In-memory caching layer
