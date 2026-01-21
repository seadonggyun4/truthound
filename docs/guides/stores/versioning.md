# Result Versioning

Track version history, compare changes, and rollback validation results.

## Overview

The `VersionedStore` wrapper adds versioning capabilities to any base store:

- Version history tracking
- Commit messages
- Rollback to previous versions
- Diff between versions
- Optimistic locking
- Automatic cleanup of old versions

## Quick Start

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig

# Create base store
base = get_store("filesystem", base_path=".truthound/store")

# Wrap with versioning
store = VersionedStore(
    base,
    VersioningConfig(max_versions=10),
)

# Save with message
store.save(result, message="Initial validation")

# Update with message
result.tags["reviewed"] = True
store.save(result, message="Marked as reviewed")

# Get version history
history = store.get_version_history(result.run_id)
for info in history:
    print(f"v{info.version}: {info.message} ({info.created_at})")

# Rollback to version 1
store.rollback(result.run_id, version=1)
```

## Configuration

```python
from truthound.stores.versioning.base import VersioningConfig, VersioningMode

config = VersioningConfig(
    mode=VersioningMode.INCREMENTAL,  # Versioning strategy
    max_versions=10,                   # Max versions to keep (0 = unlimited)
    auto_cleanup=True,                 # Auto-remove old versions
    track_changes=True,                # Store change details
    require_message=False,             # Require commit message
    enable_branching=False,            # Support version branching
    checksum_algorithm="sha256",       # Content checksum algorithm
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `VersioningMode` | `INCREMENTAL` | Version number strategy |
| `max_versions` | `int` | `0` | Max versions per item (0 = unlimited) |
| `auto_cleanup` | `bool` | `True` | Automatically remove old versions |
| `track_changes` | `bool` | `True` | Store detailed change information |
| `require_message` | `bool` | `False` | Require message on save |
| `enable_branching` | `bool` | `False` | Enable version branches |
| `checksum_algorithm` | `str` | `sha256` | Checksum algorithm (sha256, sha1, md5) |

## Versioning Strategies

Four versioning strategies are available:

### Incremental (Default)

Simple sequential integers: 1, 2, 3...

```python
from truthound.stores.versioning.strategies import IncrementalStrategy

strategy = IncrementalStrategy()
strategy.format_version(1)   # "v1"
strategy.format_version(10)  # "v10"
```

### Semantic

Semantic versioning format: X.Y.Z (major.minor.patch)

Internally stored as: `major * 10000 + minor * 100 + patch`

```python
from truthound.stores.versioning.strategies import SemanticStrategy

strategy = SemanticStrategy()

# Initial version: 1.0.0
strategy.get_next_version("item", None)  # 10000

# Bump patch: 1.0.1
strategy.get_next_version("item", 10000, {"bump": "patch"})  # 10001

# Bump minor: 1.1.0
strategy.get_next_version("item", 10001, {"bump": "minor"})  # 10100

# Bump major: 2.0.0
strategy.get_next_version("item", 10100, {"bump": "major"})  # 20000

# Format
strategy.format_version(10101)  # "1.1.1"
strategy.parse_version("2.0.0")  # 20000
```

### Timestamp

Unix timestamps in milliseconds, displayed as ISO format:

```python
from truthound.stores.versioning.strategies import TimestampStrategy

strategy = TimestampStrategy()

version = strategy.get_next_version("item", None)
# Returns current timestamp in ms, e.g., 1704067200000

strategy.format_version(version)
# "2024-01-01T00:00:00"
```

### Git-Like

Sequential integers with associated content hashes:

```python
from truthound.stores.versioning.strategies import GitLikeStrategy

strategy = GitLikeStrategy(hash_length=7)

# Generate version with content hash
version = strategy.get_next_version(
    "item",
    None,
    {"content": {"data": "test"}},
)

strategy.format_version(version)
# "abc1234" (short hash) or "0000001" if no content provided

# Get full content hash
strategy.get_content_hash(version)
# "abc1234def5678..." (full SHA-256)
```

### Using Strategies

```python
from truthound.stores.versioning.base import VersioningMode
from truthound.stores.versioning import VersionedStore, VersioningConfig

# Use semantic versioning
store = VersionedStore(
    base_store,
    VersioningConfig(mode=VersioningMode.SEMANTIC),
)

# Save with bump type
store.save(result, message="Fix validation", metadata={"bump": "patch"})
store.save(result, message="Add new checks", metadata={"bump": "minor"})
store.save(result, message="Breaking change", metadata={"bump": "major"})
```

## Core Operations

### Save with Versioning

```python
# Basic save
store.save(result)

# Save with message
store.save(result, message="Updated thresholds")

# Save with author
store.save(result, message="Fix issue", created_by="admin")

# Save with metadata
store.save(
    result,
    message="Quarterly validation",
    metadata={"quarter": "Q1", "year": 2024},
)
```

### Retrieve Specific Version

```python
# Get latest version
result = store.get(item_id)

# Get specific version
result_v1 = store.get(item_id, version=1)
result_v3 = store.get(item_id, version=3)
```

### Check Version Existence

```python
# Check if item exists (any version)
store.exists(item_id)

# Check specific version
store.exists(item_id, version=2)
```

### Delete Versions

```python
# Delete specific version
store.delete(item_id, version=2)

# Delete all versions
store.delete(item_id, delete_all_versions=True)
```

## Version History

### Get History

```python
# Get all versions (newest first)
history = store.get_version_history(item_id)

# With pagination
history = store.get_version_history(item_id, limit=5, offset=0)

for info in history:
    print(f"Version {info.version}")
    print(f"  Created: {info.created_at}")
    print(f"  By: {info.created_by}")
    print(f"  Message: {info.message}")
    print(f"  Parent: {info.parent_version}")
    print(f"  Checksum: {info.checksum}")
    print(f"  Size: {info.size_bytes} bytes")
```

### Version Info

```python
from truthound.stores.versioning.base import VersionInfo

# Get specific version info
info = store.get_version_info(item_id, version=2)

# Get latest version info
latest = store.get_latest_version_info(item_id)

# Count versions
count = store.count_versions(item_id)
```

### VersionInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Version number |
| `item_id` | `str` | Item identifier |
| `created_at` | `datetime` | Creation timestamp |
| `created_by` | `str \| None` | Author |
| `message` | `str \| None` | Commit message |
| `parent_version` | `int \| None` | Previous version |
| `metadata` | `dict[str, Any]` | Additional metadata |
| `checksum` | `str \| None` | Content checksum |
| `size_bytes` | `int` | Data size |

## Diff and Rollback

### Compare Versions

```python
# Compare two specific versions
diff = store.diff(item_id, version_a=1, version_b=3)

# Compare with latest
diff = store.diff(item_id, version_a=1)

print(f"Summary: {diff.summary}")
# "2 added, 1 removed, 3 modified"

for change in diff.changes:
    print(f"  {change['path']}: {change['type']}")
    print(f"    Old: {change['old_value']}")
    print(f"    New: {change['new_value']}")
```

### VersionDiff Structure

```python
from truthound.stores.versioning.base import VersionDiff, DiffType

diff = VersionDiff(
    item_id="run-123",
    version_a=1,
    version_b=3,
    changes=[
        {
            "path": "status",
            "type": DiffType.MODIFIED.value,
            "old_value": "success",
            "new_value": "failure",
        },
        {
            "path": "tags.reviewed",
            "type": DiffType.ADDED.value,
            "old_value": None,
            "new_value": True,
        },
    ],
    summary="1 added, 0 removed, 1 modified",
)
```

### Rollback

Rollback creates a new version that copies an old version:

```python
# Rollback to version 1
store.rollback(item_id, version=1)

# Rollback with message
store.rollback(
    item_id,
    version=2,
    message="Revert due to issue",
    created_by="admin",
)

# Check history after rollback
history = store.get_version_history(item_id)
# Latest version will have metadata: {"rollback_from": 2}
```

## Optimistic Locking

Prevent concurrent modification conflicts:

```python
from truthound.stores.versioning.base import VersionConflictError

# Get current version
latest = store.get_latest_version_info(item_id)
current_version = latest.version if latest else None

# Try to save with expected version
try:
    store.save(
        result,
        message="Update",
        expected_version=current_version,
    )
except VersionConflictError as e:
    print(f"Conflict: expected v{e.expected_version}, actual v{e.actual_version}")
    # Reload and retry
```

## Custom Version Store

By default, version metadata is stored in memory. For production, implement a persistent store:

```python
from truthound.stores.versioning.base import VersionStore, VersionInfo

class DatabaseVersionStore(VersionStore):
    """Store version info in database."""

    def __init__(self, connection_url: str):
        self._engine = create_engine(connection_url)
        # ... setup tables

    def save_version_info(self, info: VersionInfo) -> None:
        # Save to database
        ...

    def get_version_info(self, item_id: str, version: int) -> VersionInfo:
        # Load from database
        ...

    def list_versions(
        self,
        item_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        # Query database
        ...

    def get_latest_version(self, item_id: str) -> VersionInfo | None:
        # Query latest
        ...

    def delete_version(self, item_id: str, version: int) -> bool:
        # Delete from database
        ...

    def count_versions(self, item_id: str) -> int:
        # Count in database
        ...

# Use custom version store
store = VersionedStore(
    base_store,
    config,
    version_store=DatabaseVersionStore("postgresql://..."),
)
```

## Error Handling

```python
from truthound.stores.versioning.base import (
    VersioningError,
    VersionConflictError,
    VersionNotFoundError,
)

try:
    result = store.get(item_id, version=999)
except VersionNotFoundError as e:
    print(f"Version {e.version} not found for {e.item_id}")

try:
    store.save(result, expected_version=1)
except VersionConflictError as e:
    print(f"Conflict: expected {e.expected_version}, got {e.actual_version}")
```

## Composing with Other Features

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore
from truthound.stores.caching.backends import LRUCache

# Base store
base = get_store("s3", bucket="my-bucket")

# Add versioning
versioned = VersionedStore(
    base,
    VersioningConfig(max_versions=10),
)

# Add caching on top
cache = LRUCache(max_size=100)
cached = CachedStore(versioned, cache)
```

## Next Steps

- [Retention](retention.md) - TTL and retention policies
- [Caching](caching.md) - In-memory caching layer
- [Observability](observability.md) - Audit, metrics, tracing
