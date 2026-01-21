# Report Versioning

Truthound Data Docs는 리포트 버전 관리를 위한 4가지 전략을 제공합니다.

## 버저닝 전략

| Strategy | 형식 | 설명 |
|----------|------|------|
| `Incremental` | `1, 2, 3, ...` | 단순 증가 번호 |
| `Semantic` | `1.0.0, 1.1.0, 2.0.0` | 시맨틱 버저닝 (major.minor.patch) |
| `Timestamp` | `20250115_103045` | 타임스탬프 기반 |
| `GitLike` | `abc123de` | 콘텐츠 해시 기반 (8자 hex) |

## 기본 사용법

### 버전 스토리지 생성

```python
from truthound.datadocs.versioning import (
    FileVersionStorage,
    InMemoryVersionStorage,
)

# 파일 기반 스토리지
storage = FileVersionStorage(base_dir="./report_versions")

# 메모리 기반 스토리지 (테스트용)
storage = InMemoryVersionStorage()
```

### 리포트 버전 저장

```python
from truthound.datadocs.versioning import FileVersionStorage

storage = FileVersionStorage(base_dir="./report_versions")

# 새 버전 저장
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

### 버전 조회

```python
# 특정 버전 조회
report = storage.load("customer_report", version=2)
print(report.content)

# 최신 버전 조회
latest = storage.load("customer_report")  # version=None이면 최신

# 최신 버전 번호만 조회
latest_version = storage.get_latest_version("customer_report")
# 3
```

### 버전 목록

```python
# 모든 버전 목록
versions = storage.list_versions("customer_report")
for v in versions:
    print(f"v{v.version} - {v.created_at} - {v.message}")

# 페이지네이션
versions = storage.list_versions("customer_report", limit=10, offset=0)

# 버전 개수
count = storage.count_versions("customer_report")
```

### 버전 삭제

```python
# 특정 버전 삭제
success = storage.delete_version("customer_report", version=1)
```

## VersionInfo

버전 메타데이터를 담는 데이터 클래스입니다.

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

# dict로 변환
data = info.to_dict()

# dict에서 생성
info = VersionInfo.from_dict(data)
```

## ReportVersion

버전 정보와 콘텐츠를 함께 담는 클래스입니다.

```python
from truthound.datadocs.versioning import ReportVersion

report = ReportVersion(
    info=version_info,
    content=html_content,
    format="html",
)

# 속성 접근
print(report.version)   # info.version
print(report.checksum)  # SHA256 해시
```

## 버저닝 전략

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

# major 버전 증가
next_ver = strategy.next_version(
    current_version=100,  # 1.0.0
    metadata={"bump": "major"},
)
# 200 (2.0.0)

# minor 버전 증가
next_ver = strategy.next_version(
    current_version=100,
    metadata={"bump": "minor"},
)
# 110 (1.1.0)

# patch 버전 증가 (기본값)
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
# Unix timestamp (예: 1705312245)

# ISO 날짜로 포맷팅됨
```

### GitLikeStrategy

```python
from truthound.datadocs.versioning.version import GitLikeStrategy

strategy = GitLikeStrategy()
next_ver = strategy.next_version(
    current_version=None,
    metadata={"content": html_content},
)
# 콘텐츠 해시 기반 8자 hex (예: "abc123de")
```

## 버전 비교 (Diff)

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

# 변경 여부 확인
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

### Diff 전략

#### TextDiffStrategy

텍스트 기반 unified diff를 생성합니다.

```python
from truthound.datadocs.versioning.diff import TextDiffStrategy

strategy = TextDiffStrategy(
    context_lines=3,
    ignore_whitespace=False,
)
result = strategy.diff(old_content, new_content)
```

#### StructuralDiffStrategy

JSON/구조화된 데이터 비교를 수행합니다.

```python
from truthound.datadocs.versioning.diff import StructuralDiffStrategy

strategy = StructuralDiffStrategy(
    include_content=False,
    max_depth=10,
)
result = strategy.diff(old_content, new_content)
```

#### SemanticDiffStrategy

의미 있는 변경 사항을 추출합니다.

```python
from truthound.datadocs.versioning.diff import SemanticDiffStrategy

strategy = SemanticDiffStrategy()
result = strategy.diff(old_content, new_content)
```

### ReportDiffer

고수준 diff API를 제공합니다.

```python
from truthound.datadocs.versioning.diff import ReportDiffer, diff_versions

differ = ReportDiffer()

# 두 버전 비교
result = differ.compare(old_version, new_version)

# 특정 전략으로 비교
result = differ.compare_with_strategy(
    old_version,
    new_version,
    strategy="text",  # "text", "structural", "semantic"
)

# diff 결과 포맷팅
formatted = differ.format_diff(result, format="unified")
# format: "unified", "summary", "json"

# 간편 함수
result = diff_versions(old_version, new_version, strategy="text")
```

## 스토리지 백엔드

### FileVersionStorage

파일 시스템 기반 스토리지입니다.

```
base_dir/
├── customer_report/
│   ├── versions.json    # 메타데이터
│   ├── v1.html          # 버전 1 콘텐츠
│   ├── v2.html          # 버전 2 콘텐츠
│   └── v3.html          # 버전 3 콘텐츠
└── sales_report/
    ├── versions.json
    └── v1.html
```

```python
from truthound.datadocs.versioning import FileVersionStorage

storage = FileVersionStorage(
    base_dir="./report_versions",
)

# 모든 VersionStorage 메서드 지원
storage.save(...)
storage.load(...)
storage.list_versions(...)
```

### InMemoryVersionStorage

메모리 기반 스토리지로, 테스트 및 개발용입니다.

```python
from truthound.datadocs.versioning import InMemoryVersionStorage

storage = InMemoryVersionStorage()

# 프로그램 종료 시 데이터 손실
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
        """새 버전 저장."""
        ...

    @abstractmethod
    def load(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportVersion | None:
        """버전 로드 (None이면 최신)."""
        ...

    @abstractmethod
    def list_versions(
        self,
        report_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """버전 목록 조회."""
        ...

    @abstractmethod
    def get_latest_version(self, report_id: str) -> int | None:
        """최신 버전 번호 조회."""
        ...

    @abstractmethod
    def delete_version(self, report_id: str, version: int) -> bool:
        """버전 삭제."""
        ...

    @abstractmethod
    def count_versions(self, report_id: str) -> int:
        """버전 개수 조회."""
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
    def checksum(self) -> str: ...  # SHA256 해시
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

- [HTML Reports](html-reports.md) - HTML 리포트 생성
- [PDF Export](pdf-export.md) - PDF 내보내기
