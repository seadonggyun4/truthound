# Report Versioning

Truthound Data Docs provides 4 strategies for report version management.

## Versioning Strategies

| Strategy | Format | Description |
|----------|--------|-------------|
| `Incremental` | `1, 2, 3, ...` | Simple incrementing number |
| `Semantic` | `1.0.0, 1.1.0, 2.0.0` | Semantic versioning (major.minor.patch) |
| `Timestamp` | `20250115_103045` | Timestamp-based |
| `GitLike` | `abc123de` | Content hash-based (8-character hex) |

## Basic Usage

### Creating Version Storage

```python
from truthound.datadocs.versioning import (
    FileVersionStorage,
    InMemoryVersionStorage,
)

# File-based storage
storage = FileVersionStorage(base_dir="./report_versions")

# Memory-based storage (for testing)
storage = InMemoryVersionStorage()
```

### Saving Report Versions

```python
from truthound.datadocs.versioning import FileVersionStorage

storage = FileVersionStorage(base_dir="./report_versions")

# Save new version
version = storage.save(
    report_id="customer_report",
    content=html_content,
    format="html",
    message="Added new quality metrics",
    created_by="data-team",
    metadata={"title": "Customer Data", "theme": "professional"},
)

print(f"Saved version: {version.info.version}")
# Saved version: 1
```

### Retrieving Versions

```python
# Retrieve specific version
report = storage.load("customer_report", version=2)
print(report.content)

# Retrieve latest version
latest = storage.load("customer_report")  # version=None returns latest

# Get latest version number only
latest_version = storage.get_latest_version("customer_report")
# 3
```

### Listing Versions

```python
# List all versions
versions = storage.list_versions("customer_report")
for v in versions:
    print(f"v{v.version} - {v.created_at} - {v.message}")

# Pagination
versions = storage.list_versions("customer_report", limit=10, offset=0)

# Version count
count = storage.count_versions("customer_report")
```

### Deleting Versions

```python
# Delete specific version
success = storage.delete_version("customer_report", version=1)
```

## VersionInfo

A data class containing version metadata.

```python
from truthound.datadocs.versioning import VersionInfo
from datetime import datetime

info = VersionInfo(
    version=1,
    report_id="customer_report",
    created_at=datetime.now(),
    created_by="data-team",
    message="Initial report",
    parent_version=None,
    checksum="abc123...",
    size_bytes=12345,
    metadata={"title": "Customer Data"},
)

# Convert to dict
data = info.to_dict()

# Create from dict
info = VersionInfo.from_dict(data)
```

## ReportVersion

A class containing version information and content together.

```python
from truthound.datadocs.versioning import ReportVersion

report = ReportVersion(
    info=version_info,
    content=html_content,
    format="html",
)

# Property access
print(report.version)   # info.version
print(report.checksum)  # SHA256 hash
```

## Versioning Strategies

### IncrementalStrategy

```python
from truthound.datadocs.versioning.version import IncrementalStrategy

strategy = IncrementalStrategy()
next_ver = strategy.next_version(current_version=5, metadata=None)
# 6
```

### SemanticStrategy

```python
from truthound.datadocs.versioning.version import SemanticStrategy

strategy = SemanticStrategy()

# Major version bump
next_ver = strategy.next_version(
    current_version=100,  # 1.0.0
    metadata={"bump": "major"},
)
# 200 (2.0.0)

# Minor version bump
next_ver = strategy.next_version(
    current_version=100,
    metadata={"bump": "minor"},
)
# 110 (1.1.0)

# Patch version bump (default)
next_ver = strategy.next_version(
    current_version=100,
    metadata={"bump": "patch"},
)
# 101 (1.0.1)
```

### TimestampStrategy

```python
from truthound.datadocs.versioning.version import TimestampStrategy

strategy = TimestampStrategy()
next_ver = strategy.next_version(current_version=None, metadata=None)
# Unix timestamp (e.g., 1705312245)

# Formatted as ISO date
```

### GitLikeStrategy

```python
from truthound.datadocs.versioning.version import GitLikeStrategy

strategy = GitLikeStrategy()
next_ver = strategy.next_version(
    current_version=None,
    metadata={"content": html_content},
)
# Content hash-based 8-character hex (e.g., "abc123de")
```

## Version Comparison (Diff)

### DiffResult

```python
from truthound.datadocs.versioning.diff import DiffResult, ChangeType

result = DiffResult(
    old_version=1,
    new_version=2,
    report_id="customer_report",
    changes=[...],
    summary={"added": 5, "removed": 2, "modified": 3, "unchanged": 10},
    unified_diff="...",
)

# Check for changes
if result.has_changes():
    print(f"Added: {result.added_count}")
    print(f"Removed: {result.removed_count}")
    print(f"Modified: {result.modified_count}")
```

### Change

```python
from truthound.datadocs.versioning.diff import Change, ChangeType

change = Change(
    change_type=ChangeType.MODIFIED,
    path="sections.overview.metrics.row_count",
    old_value=1000,
    new_value=1500,
    line_number=42,
)
```

### Diff Strategies

#### TextDiffStrategy

Generates text-based unified diff.

```python
from truthound.datadocs.versioning.diff import TextDiffStrategy

strategy = TextDiffStrategy(
    context_lines=3,
    ignore_whitespace=False,
)
result = strategy.diff(old_content, new_content)
```

#### StructuralDiffStrategy

Performs JSON/structured data comparison.

```python
from truthound.datadocs.versioning.diff import StructuralDiffStrategy

strategy = StructuralDiffStrategy(
    include_content=False,
    max_depth=10,
)
result = strategy.diff(old_content, new_content)
```

#### SemanticDiffStrategy

Extracts meaningful changes.

```python
from truthound.datadocs.versioning.diff import SemanticDiffStrategy

strategy = SemanticDiffStrategy()
result = strategy.diff(old_content, new_content)
```

### ReportDiffer

Provides a high-level diff API.

```python
from truthound.datadocs.versioning.diff import ReportDiffer, diff_versions

differ = ReportDiffer()

# Compare two versions
result = differ.compare(old_version, new_version)

# Compare with specific strategy
result = differ.compare_with_strategy(
    old_version,
    new_version,
    strategy="text",  # "text", "structural", "semantic"
)

# Format diff result
formatted = differ.format_diff(result, format="unified")
# format: "unified", "summary", "json"

# Convenience function
result = diff_versions(old_version, new_version, strategy="text")
```

## Storage Backends

### FileVersionStorage

File system-based storage.

```
base_dir/
├── customer_report/
│   ├── versions.json    # Metadata
│   ├── v1.html          # Version 1 content
│   ├── v2.html          # Version 2 content
│   └── v3.html          # Version 3 content
└── sales_report/
    ├── versions.json
    └── v1.html
```

```python
from truthound.datadocs.versioning import FileVersionStorage

storage = FileVersionStorage(
    base_dir="./report_versions",
)

# Supports all VersionStorage methods
storage.save(...)
storage.load(...)
storage.list_versions(...)
```

### InMemoryVersionStorage

Memory-based storage for testing and development.

```python
from truthound.datadocs.versioning import InMemoryVersionStorage

storage = InMemoryVersionStorage()

# Data is lost when program terminates
```

## API Reference

### VersionStorage (Abstract Base)

```python
class VersionStorage(ABC):
    @abstractmethod
    def save(
        self,
        report_id: str,
        content: bytes | str,
        format: str = "html",
        message: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReportVersion:
        """Save new version."""
        ...

    @abstractmethod
    def load(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportVersion | None:
        """Load version (None returns latest)."""
        ...

    @abstractmethod
    def list_versions(
        self,
        report_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """List versions."""
        ...

    @abstractmethod
    def get_latest_version(self, report_id: str) -> int | None:
        """Get latest version number."""
        ...

    @abstractmethod
    def delete_version(self, report_id: str, version: int) -> bool:
        """Delete version."""
        ...

    @abstractmethod
    def count_versions(self, report_id: str) -> int:
        """Get version count."""
        ...
```

### VersionInfo

```python
@dataclass
class VersionInfo:
    version: int
    report_id: str
    created_at: datetime
    created_by: str | None = None
    message: str | None = None
    parent_version: int | None = None
    checksum: str | None = None
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo": ...
```

### ReportVersion

```python
@dataclass
class ReportVersion:
    info: VersionInfo
    content: bytes | str
    format: str = "html"

    @property
    def version(self) -> int: ...
    @property
    def checksum(self) -> str: ...  # SHA256 hash
```

### ChangeType

```python
class ChangeType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
```

## See Also

- [HTML Reports](html-reports.md) - HTML report generation
- [PDF Export](pdf-export.md) - PDF export
