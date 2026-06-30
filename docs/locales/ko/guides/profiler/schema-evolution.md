# 스키마 Evolution

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Change, Detection, Identify을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 History, Management, Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Continuous, Monitoring, Watch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Impact, Analysis, Assess을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Advanced, Rename, Detection, Multiple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설치

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

```bash
pip install truthound
```

## 빠른 시작

### Detect 스키마 Changes

```python
from truthound.profiler.evolution import SchemaEvolutionDetector

# Create detector
detector = SchemaEvolutionDetector()

# Define schemas
old_schema = {"id": "Int64", "name": "Utf8", "email": "Utf8"}
new_schema = {"id": "Int64", "name": "Utf8", "user_email": "Utf8", "age": "Int32"}

# Detect changes
changes = detector.detect_changes(new_schema, old_schema)

for change in changes:
    print(f"[{change.severity.value}] {change.description}")
    if change.breaking:
        print(f"  ⚠️ BREAKING CHANGE")
    if change.migration_hint:
        print(f"  💡 Hint: {change.migration_hint}")
```

실무 운영 가이드에서 Output을(를) 다루는 항목입니다:
```
[critical] Column 'email' removed (was type Utf8)
  ⚠️ BREAKING CHANGE
  💡 Hint: Column removed - update consumers to not depend on this column
[info] Column 'user_email' added with type Utf8
  💡 Hint: New column - ensure consumers handle missing data for older records
[info] Column 'age' added with type Int32
  💡 Hint: New column - ensure consumers handle missing data for older records
```

### Manage 스키마 History

```python
from truthound.profiler.evolution import SchemaHistory

# Create history with semantic versioning
history = SchemaHistory.create(
    storage_type="file",
    path="./schema_history",
    version_strategy="semantic",
)

# Save initial schema
v1 = history.save(
    {"id": "Int64", "name": "Utf8"},
    metadata={"author": "data-team"},
)
print(f"Version: {v1.version}")  # 1.0.0

# Save updated schema (non-breaking addition)
v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})
print(f"Version: {v2.version}")  # 1.1.0 (minor bump)

# Get diff between versions
diff = history.diff(v1, v2)
print(diff.format_text())

# Rollback if needed
v3 = history.rollback(v1, reason="Incompatible with legacy systems")
```

### Watch for Changes

```python
from truthound.profiler.evolution import SchemaWatcher, FileSchemaSource

# Create watcher
watcher = SchemaWatcher()

# Add schema files to watch
watcher.add_source(FileSchemaSource("./schemas/users.json"))
watcher.add_source(FileSchemaSource("./schemas/orders.json"))

# Add event handler
def on_change(event):
    if event.has_breaking_changes():
        print(f"🚨 Breaking changes in {event.source}!")
        for change in event.changes:
            if change.breaking:
                print(f"  - {change.description}")

from truthound.profiler.evolution import CallbackEventHandler
watcher.add_handler(CallbackEventHandler(on_change))

# Start watching (polls every 60 seconds)
watcher.start(poll_interval=60)

# ... later ...
watcher.stop()
```

## CLI Commands

### 스키마 Check

실무 운영 가이드에서 Compare을(를) 다루는 항목입니다:

```bash
# Compare current schema to baseline
th schema-check current.json --baseline baseline.json

# Output as JSON
th schema-check current.json -b baseline.json --format json

# Output as Markdown (for reports)
th schema-check current.json -b baseline.json --format markdown

# Fail CI/CD on breaking changes
th schema-check current.json -b baseline.json --fail-on-breaking

# Disable rename detection
th schema-check current.json -b baseline.json --no-detect-renames

# Adjust similarity threshold for rename detection
th schema-check current.json -b baseline.json --similarity 0.7
```

### 스키마 History

Manage 스키마 version history:

```bash
# Initialize history storage
th schema-history init ./my_history
th schema-history init ./my_history --strategy timestamp

# Save a schema version
th schema-history save schema.json
th schema-history save schema.json --version 2.0.0
th schema-history save schema.json -m "Added email column"

# List versions
th schema-history list
th schema-history list --limit 20
th schema-history list --format json

# Show version details
th schema-history show 1.0.0
th schema-history show abc12345 --format json

# Rollback to a version
th schema-history rollback 1.0.0
th schema-history rollback 1.0.0 --reason "Incompatible change" --yes
```

### 스키마 Diff

실무 운영 가이드에서 Show을(를) 다루는 항목입니다:

```bash
# Diff between two versions
th schema-diff 1.0.0 2.0.0

# Diff to latest
th schema-diff 1.0.0

# Output formats
th schema-diff 1.0.0 2.0.0 --format json
th schema-diff 1.0.0 2.0.0 --format markdown
```

### 스키마 Watch

실무 운영 가이드에서 Monitor을(를) 다루는 항목입니다:

```bash
# Watch a single file
th schema-watch schema.json

# Watch multiple files
th schema-watch schema1.json schema2.json

# Custom poll interval
th schema-watch schema.json --interval 30

# Track history
th schema-watch schema.json --history ./schema_history

# Write alerts to file
th schema-watch schema.json --alert-file alerts.jsonl

# Only alert on breaking changes
th schema-watch schema.json --only-breaking

# Single check (for CI/CD)
th schema-watch schema.json --once --only-breaking
```

## Python API

### SchemaEvolutionDetector

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound.profiler.evolution import SchemaEvolutionDetector

detector = SchemaEvolutionDetector(
    detect_renames=True,           # Enable rename detection
    rename_similarity_threshold=0.8,  # Threshold for considering a rename
)

# Can accept various schema formats
# - dict: {"col": "Type"}
# - polars.Schema
# - polars.DataFrame
# - polars.LazyFrame

changes = detector.detect_changes(current_schema, baseline_schema)
summary = detector.get_change_summary(changes)

print(f"Total changes: {summary.total_changes}")
print(f"Breaking changes: {summary.breaking_changes}")
print(f"Compatibility: {summary.compatibility_level.value}")
```

### Change Types

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Change, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Breaking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------------|-----------|----------|
| 실무 운영 가이드에서 `COLUMN_ADDED`, COLUMN_ADDED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLUMN_REMOVED`, COLUMN_REMOVED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLUMN_RENAMED`, COLUMN_RENAMED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TYPE_CHANGED`, TYPE_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Depends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 WARNING/CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NULLABLE_CHANGED`, NULLABLE_CHANGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Depends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 INFO/CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Type, Int32, Int64, Nullable, NonNullable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### SchemaHistory

실무 운영 가이드에서 Manage을(를) 다루는 항목입니다:

```python
from truthound.profiler.evolution import SchemaHistory

# Create with different strategies
history = SchemaHistory.create(
    storage_type="file",       # "file" or "memory"
    path="./schema_history",   # Required for "file"
    max_versions=100,          # Max versions to keep
    version_strategy="semantic",  # "semantic", "incremental", "timestamp", "git"
    compress=True,             # Compress stored files
)

# Save versions
v1 = history.save(schema, version="1.0.0", metadata={"author": "team"})

# Get versions
latest = history.latest
baseline = history.baseline
specific = history.get("version-id")
by_string = history.get_by_version("1.0.0")

# List and filter
versions = history.list(limit=10)
recent = history.list(since=datetime.now() - timedelta(days=7))

# Diff
diff = history.diff("1.0.0", "2.0.0")
print(diff.format_text())

# Check breaking changes
if history.has_breaking_changes_since("1.0.0"):
    print("Breaking changes detected!")

# Rollback
new_version = history.rollback("1.0.0", reason="Revert breaking change")
```

### Version Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|---------|----------|
| 실무 운영 가이드에서 `semantic`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `incremental`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `timestamp`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Time-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `git`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Git-like을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### ColumnRenameDetector

실무 운영 가이드에서 Advanced을(를) 다루는 항목입니다:

```python
from truthound.profiler.evolution import (
    ColumnRenameDetector,
    CompositeSimilarity,
    LevenshteinSimilarity,
    JaroWinklerSimilarity,
    TokenSimilarity,
)

# Use composite similarity (default)
detector = ColumnRenameDetector(
    similarity_threshold=0.8,
    require_type_match=True,
    allow_compatible_types=True,
)

result = detector.detect(
    added_columns={"user_email": "Utf8", "customer_name": "Utf8"},
    removed_columns={"email": "Utf8", "cust_name": "Utf8"},
)

# Confirmed renames (high confidence)
for rename in result.confirmed_renames:
    print(f"✓ {rename.old_name} → {rename.new_name}")
    print(f"  Similarity: {rename.similarity:.0%}")
    print(f"  Confidence: {rename.confidence.value}")

# Possible renames (need review)
for rename in result.possible_renames:
    print(f"? {rename.old_name} → {rename.new_name}")
    print(f"  Reasons: {', '.join(rename.reasons)}")

# Unmatched columns
print(f"Unmatched added: {result.unmatched_added}")
print(f"Unmatched removed: {result.unmatched_removed}")
```

실무 운영 가이드에서 Similarity을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|----------|
| 실무 운영 가이드에서 `LevenshteinSimilarity`, LevenshteinSimilarity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `JaroWinklerSimilarity`, JaroWinklerSimilarity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NgramSimilarity`, NgramSimilarity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Partial을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TokenSimilarity`, TokenSimilarity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CompositeSimilarity`, CompositeSimilarity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Weighted을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SchemaWatcher

Continuous 모니터링:

```python
from truthound.profiler.evolution import (
    SchemaWatcher,
    FileSchemaSource,
    PolarsSchemaSource,
    DictSchemaSource,
    LoggingEventHandler,
    AlertingEventHandler,
    HistoryEventHandler,
)

# Create watcher
watcher = SchemaWatcher()

# Add sources
watcher.add_source(FileSchemaSource("schema.json"))
watcher.add_source(PolarsSchemaSource(lambda: pl.read_csv("data.csv"), "data"))
watcher.add_source(DictSchemaSource({"id": "Int64"}, "manual"))

# Add handlers
watcher.add_handler(LoggingEventHandler())
watcher.add_handler(HistoryEventHandler(history))
watcher.add_handler(AlertingEventHandler(alert_manager))

# Lifecycle control
watcher.start(poll_interval=60, daemon=True)
watcher.pause()
watcher.resume()
watcher.stop()

# Check immediately
events = watcher.check_now()
```

### Breaking Change 알림

실무 운영 가이드에서 Enhanced을(를) 다루는 항목입니다:

```python
from truthound.profiler.evolution import (
    BreakingChangeAlertManager,
    ImpactAnalyzer,
    ColumnRemovedChange,
)

# Setup impact analyzer with consumer mappings
analyzer = ImpactAnalyzer()
analyzer.register_consumer("dashboard", ["users", "orders"])
analyzer.register_consumer("reports", ["users", "products"])
analyzer.register_query("users", "SELECT email FROM users")

# Create alert manager
manager = BreakingChangeAlertManager(
    impact_analyzer=analyzer,
    alert_storage_path="./alerts.json",
)

# Create alert
changes = [ColumnRemovedChange("email", "Utf8")]
alert = manager.create_alert(changes, source="users")

print(f"Alert: {alert.title}")
print(f"Impact Scope: {alert.impact.scope.value}")
print(f"Affected Consumers: {alert.impact.affected_consumers}")
print(f"Risk Level: {alert.impact.data_risk_level}/5")
print(f"Recommendations:")
for rec in alert.impact.recommendations:
    print(f"  - {rec}")

# Track alerts
history = manager.get_alert_history(status="open")
manager.acknowledge_alert(alert.alert_id)
manager.resolve_alert(alert.alert_id)

# Get statistics
stats = manager.get_stats()
print(f"Total: {stats['total']}, Open: {stats['open']}")
```

Format 알림 for notifications:

```python
# Slack
slack_payload = alert.format_slack_message()

# Email
subject, html_body = alert.format_email()
```

## Factory Functions

실무 운영 가이드에서 Create을(를) 다루는 항목입니다:

```python
from truthound.profiler.evolution import (
    create_watcher,
    create_rename_detector,
)

# Create watcher with common settings
watcher = create_watcher(
    sources=[FileSchemaSource("schema.json")],
    poll_interval=60,
    enable_logging=True,
    enable_history=True,
    history_path="./history",
    on_change=lambda e: print(f"Change: {e.source}"),
    auto_start=True,
)

# Create rename detector with specific algorithm
detector = create_rename_detector(
    algorithm="jaro_winkler",
    threshold=0.85,
    require_type_match=True,
)
```

## 통합 예시

### CI/CD 파이프라인

```yaml
# GitHub Actions
- name: Check Schema Changes
  run: |
    th schema-check current.json -b baseline.json --fail-on-breaking
```

```python
# Python script for CI
from truthound.profiler.evolution import SchemaEvolutionDetector
import sys

detector = SchemaEvolutionDetector()
changes = detector.detect_changes(current, baseline)

breaking = [c for c in changes if c.breaking]
if breaking:
    print("❌ Breaking changes detected:")
    for c in breaking:
        print(f"  - {c.description}")
    sys.exit(1)

print("✅ No breaking changes")
```

### Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def check_schema():
    from truthound.profiler.evolution import SchemaHistory

    history = SchemaHistory.create(storage_type="file", path="/data/history")

    # Get current schema from database
    current = get_database_schema()

    # Save and check for breaking changes
    version = history.save(current)

    if version.has_breaking_changes():
        raise ValueError(f"Breaking changes: {version.changes_from_parent}")

with DAG("schema_monitor", schedule_interval="@daily") as dag:
    check_schema_task = PythonOperator(
        task_id="check_schema",
        python_callable=check_schema,
    )
```

### Slack 알림

```python
import requests
from truthound.profiler.evolution import (
    SchemaWatcher,
    BreakingChangeAlertManager,
    AlertingEventHandler,
)

class SlackNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send(self, alert):
        payload = alert.format_slack_message()
        requests.post(self.webhook_url, json=payload)
        return True

# Setup
manager = BreakingChangeAlertManager(notifiers=[
    SlackNotifier("https://hooks.slack.com/services/...")
])

watcher = SchemaWatcher()
watcher.add_handler(AlertingEventHandler(manager))
watcher.start()
```

## 권장 방식

### 1. Version Strategy Selection

- 실무 운영 가이드에서 API, Semantic, Best, APIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Timestamp, Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Git-like, Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 2. Breaking Change Policy

```python
# Define what constitutes breaking
BREAKING_TYPES = {
    ChangeType.COLUMN_REMOVED,
    ChangeType.TYPE_CHANGED,  # Only incompatible
    ChangeType.NULLABLE_CHANGED,  # Only nullable→non-nullable
}

# Allow grace period for breaking changes
MIN_DEPRECATION_DAYS = 30
```

### 3. Rename Detection Tuning

```python
# Strict (production)
detector = ColumnRenameDetector(
    similarity_threshold=0.9,
    require_type_match=True,
)

# Lenient (development)
detector = ColumnRenameDetector(
    similarity_threshold=0.7,
    require_type_match=False,
)
```

### 4. 알림 설정

```python
# Critical alerts only
watcher.add_handler(AlertingEventHandler(
    alert_manager,
    only_breaking=True,
))

# All changes with logging
watcher.add_handler(LoggingEventHandler(
    min_severity=ChangeSeverity.INFO,
))
```

## Related 가이드

- 실무 운영 가이드에서 Drift, Detection, Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Quality, Scoring, Impact을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [체크포인트 Basics](../checkpoint/basics.md) - CI/CD 통합
- 실무 운영 가이드에서 Notifications, Alert을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 레퍼런스

실무 운영 가이드에서 API, Full을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `truthound.profiler.evolution.SchemaEvolutionDetector`, SchemaEvolutionDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.profiler.evolution.SchemaHistory`, SchemaHistory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.profiler.evolution.SchemaWatcher`, SchemaWatcher을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.profiler.evolution.ColumnRenameDetector`, ColumnRenameDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.profiler.evolution.BreakingChangeAlertManager`, BreakingChangeAlertManager을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `truthound.profiler.evolution.ImpactAnalyzer`, ImpactAnalyzer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
