# Drift Detection

이 문서는 스키마 변화 및 데이터 드리프트 감지 시스템을 설명합니다.

## 개요

`src/truthound/profiler/evolution/detector.py`에 구현된 드리프트 감지 시스템은 시간에 따른 스키마 및 데이터 변화를 추적합니다.

## SchemaChangeType

```python
class SchemaChangeType(str, Enum):
    """스키마 변경 유형"""

    COLUMN_ADDED = "column_added"       # 새 컬럼 추가
    COLUMN_REMOVED = "column_removed"   # 컬럼 삭제
    COLUMN_RENAMED = "column_renamed"   # 컬럼 이름 변경
    TYPE_CHANGED = "type_changed"       # 데이터 타입 변경
```

## SchemaChange

```python
@dataclass
class SchemaChange:
    """스키마 변경 정보"""

    change_type: SchemaChangeType
    column_name: str
    old_value: Any = None   # 이전 값 (타입, 이름 등)
    new_value: Any = None   # 새로운 값
    severity: str = "medium"
    description: str = ""
```

## SchemaChangeDetector Protocol

```python
from typing import Protocol

class SchemaChangeDetector(Protocol):
    """스키마 변경 감지기 프로토콜"""

    def detect_changes(
        self,
        old_profile: TableProfile,
        new_profile: TableProfile,
    ) -> list[SchemaChange]:
        """두 프로파일 간 스키마 변경 감지"""
        ...
```

## 타입 호환성 매핑

안전한 타입 업그레이드를 정의합니다.

```python
# 호환 가능한 타입 변환 (safe upgrade)
TYPE_COMPATIBILITY = {
    "Int8": ["Int16", "Int32", "Int64", "Float32", "Float64"],
    "Int16": ["Int32", "Int64", "Float32", "Float64"],
    "Int32": ["Int64", "Float64"],
    "Int64": ["Float64"],
    "Float32": ["Float64"],
    "Utf8": ["LargeUtf8"],
}

def is_compatible_change(old_type: str, new_type: str) -> bool:
    """타입 변경이 호환 가능한지 확인"""
    return new_type in TYPE_COMPATIBILITY.get(old_type, [])
```

## 기본 사용법

```python
from truthound.profiler.evolution import SchemaEvolutionDetector

detector = SchemaEvolutionDetector()

# 스키마 변경 감지
changes = detector.detect_changes(old_profile, new_profile)

for change in changes:
    print(f"Type: {change.change_type}")
    print(f"Column: {change.column_name}")
    print(f"Severity: {change.severity}")
    if change.change_type == SchemaChangeType.TYPE_CHANGED:
        print(f"  {change.old_value} -> {change.new_value}")
```

## 컬럼 이름 변경 감지

유사한 통계를 가진 컬럼을 분석하여 이름 변경을 추론합니다.

```python
from truthound.profiler.evolution import ColumnRenameDetector

detector = ColumnRenameDetector(
    similarity_threshold=0.9,  # 90% 이상 유사도
)

renames = detector.detect_renames(old_profile, new_profile)

for rename in renames:
    print(f"Rename detected: {rename.old_name} -> {rename.new_name}")
    print(f"Confidence: {rename.confidence:.2%}")
```

## 호환성 분석

```python
from truthound.profiler.evolution import CompatibilityAnalyzer

analyzer = CompatibilityAnalyzer()

report = analyzer.analyze(old_profile, new_profile)

print(f"Compatible: {report.is_compatible}")
print(f"Breaking changes: {len(report.breaking_changes)}")
print(f"Warnings: {len(report.warnings)}")

for breaking in report.breaking_changes:
    print(f"  BREAKING: {breaking.description}")
```

## 드리프트 심각도

| 심각도 | 설명 | 예시 |
|--------|------|------|
| `info` | 정보성 변경 | 새 컬럼 추가 |
| `low` | 사소한 변경 | 호환 가능한 타입 확장 |
| `medium` | 주의 필요 | 컬럼 이름 변경 |
| `high` | 조사 필요 | 타입 변경 (비호환) |
| `critical` | 즉시 조치 필요 | 필수 컬럼 삭제 |

## Breaking Change 알림

```python
from truthound.profiler.evolution import BreakingChangeAlert

alerts = detector.get_breaking_alerts(changes)

for alert in alerts:
    print(f"ALERT: {alert.message}")
    print(f"Impact: {alert.impact}")
    print(f"Recommendation: {alert.recommendation}")
```

## 히스토리 추적

```python
from truthound.profiler.evolution import SchemaHistory

history = SchemaHistory(storage_dir=".truthound/schema_history")

# 프로파일 저장
history.save(profile, version="v1.0")
history.save(new_profile, version="v1.1")

# 히스토리 조회
versions = history.list_versions()

# 버전 간 비교
changes = history.compare("v1.0", "v1.1")

# 특정 버전 로드
old_profile = history.load("v1.0")
```

## 자동 알림

```python
from truthound.profiler.evolution import SchemaWatcher

watcher = SchemaWatcher(
    alert_callback=lambda alert: send_slack_notification(alert),
    check_interval_minutes=60,
)

# 모니터링 시작
watcher.watch("data.csv", baseline_profile)

# 변경 감지 시 자동 알림 전송
```

## CLI 사용법

```bash
# 두 프로파일 비교
th compare profile_v1.json profile_v2.json

# 스키마 변경 감지
th schema-diff old_profile.json new_profile.json

# 호환성 분석
th check-compatibility old_profile.json new_profile.json

# Breaking change 확인
th check-breaking old_profile.json new_profile.json
```

## 통합 예제

```python
from truthound.profiler import TableProfiler
from truthound.profiler.evolution import SchemaEvolutionDetector
from truthound.profiler.caching import ProfileCache

# 프로파일러 및 캐시 설정
profiler = TableProfiler()
cache = ProfileCache()
detector = SchemaEvolutionDetector()

# 기준 프로파일 (캐시에서 로드 또는 생성)
baseline_key = cache.compute_fingerprint("data_baseline.csv")
baseline = cache.get_or_compute(
    baseline_key,
    lambda: profiler.profile_file("data_baseline.csv"),
)

# 현재 프로파일
current = profiler.profile_file("data_current.csv")

# 변경 감지
changes = detector.detect_changes(baseline, current)

if changes:
    print(f"Found {len(changes)} schema changes:")
    for change in changes:
        print(f"  - {change.change_type}: {change.column_name}")

    # Breaking change 확인
    breaking = [c for c in changes if c.severity == "critical"]
    if breaking:
        raise ValueError(f"Breaking changes detected: {breaking}")
```

## 다음 단계

- [품질 스코어링](quality-scoring.md) - 드리프트가 품질에 미치는 영향
- [시각화](visualization.md) - 드리프트 리포트 생성
