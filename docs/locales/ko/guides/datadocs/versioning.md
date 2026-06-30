# 리포트 버전 관리

실무 운영 가이드에서 Data Docs, Truthound, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 버전 관리 Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|--------|-------------|
| 실무 운영 가이드에서 `Incremental`, Incremental을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `1, 2, 3, ...`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `Semantic`, Semantic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `1.0.0, 1.1.0, 2.0.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Semantic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `Timestamp`, Timestamp을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `20250115_103045`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Timestamp-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `GitLike`, GitLike을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `abc123de`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Content을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

### Saving 리포트 Versions

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## 버전 관리 Strategies

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

실무 운영 가이드에서 Generates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.versioning.diff import TextDiffStrategy

strategy = TextDiffStrategy(
    context_lines=3,
    ignore_whitespace=False,
)
result = strategy.diff(old_content, new_content)
```

#### StructuralDiffStrategy

실무 운영 가이드에서 JSON, Performs, JSON/structured을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.versioning.diff import StructuralDiffStrategy

strategy = StructuralDiffStrategy(
    include_content=False,
    max_depth=10,
)
result = strategy.diff(old_content, new_content)
```

#### SemanticDiffStrategy

실무 운영 가이드에서 Extracts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.versioning.diff import SemanticDiffStrategy

strategy = SemanticDiffStrategy()
result = strategy.diff(old_content, new_content)
```

### ReportDiffer

실무 운영 가이드에서 API, Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

파일 system-based storage.

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

실무 운영 가이드에서 Memory-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.datadocs.versioning import InMemoryVersionStorage

storage = InMemoryVersionStorage()

# Data is lost when program terminates
```

## API 레퍼런스

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

## 함께 보기

- 실무 운영 가이드에서 HTML, Reports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 PDF, Export을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
