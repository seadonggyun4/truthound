# Schema Evolution

This guide covers the comprehensive schema evolution system for detecting, tracking, and managing schema changes over time.

## Overview

The schema evolution module provides:

- **Change Detection**: Identify schema differences (columns added/removed, type changes, renames)
- **History Management**: Version schemas with semantic versioning, diff, and rollback
- **Continuous Monitoring**: Watch schemas for changes with real-time alerts
- **Impact Analysis**: Assess the impact of breaking changes on downstream systems
- **Advanced Rename Detection**: Multiple similarity algorithms for detecting column renames

## Installation

The schema evolution module is included in the core Truthound package:

```bash
pip install truthound
```

## Quick Start

### Detect Schema Changes

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

Output:
```
[critical] Column 'email' removed (was type Utf8)
  ⚠️ BREAKING CHANGE
  💡 Hint: Column removed - update consumers to not depend on this column
[info] Column 'user_email' added with type Utf8
  💡 Hint: New column - ensure consumers handle missing data for older records
[info] Column 'age' added with type Int32
  💡 Hint: New column - ensure consumers handle missing data for older records
```

### Manage Schema History

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

### Schema Check

Compare schemas and detect changes:

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

### Schema History

Manage schema version history:

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

### Schema Diff

Show diff between schema versions:

```bash
# Diff between two versions
th schema-diff 1.0.0 2.0.0

# Diff to latest
th schema-diff 1.0.0

# Output formats
th schema-diff 1.0.0 2.0.0 --format json
th schema-diff 1.0.0 2.0.0 --format markdown
```

### Schema Watch

Monitor schemas for changes:

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

The core class for detecting schema changes:

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

The system detects these change types:

| Change Type | Breaking? | Severity |
|-------------|-----------|----------|
| `COLUMN_ADDED` | No | INFO |
| `COLUMN_REMOVED` | Yes | CRITICAL |
| `COLUMN_RENAMED` | Yes | CRITICAL |
| `TYPE_CHANGED` | Depends* | WARNING/CRITICAL |
| `NULLABLE_CHANGED` | Depends* | INFO/CRITICAL |

*Type changes are non-breaking for compatible upgrades (e.g., Int32 → Int64). Nullable→NonNullable is breaking.

### SchemaHistory

Manage schema versions with history:

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

| Strategy | Example | Use Case |
|----------|---------|----------|
| `semantic` | 1.2.3 | Standard versioning, auto-bumps based on change type |
| `incremental` | 1, 2, 3 | Simple incrementing numbers |
| `timestamp` | 20260128.143052 | Time-based versions |
| `git` | a1b2c3d4 | Git-like short hashes |

### ColumnRenameDetector

Advanced rename detection with multiple algorithms:

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

Similarity algorithms:

| Algorithm | Best For |
|-----------|----------|
| `LevenshteinSimilarity` | General edit distance |
| `JaroWinklerSimilarity` | Short strings, common prefixes |
| `NgramSimilarity` | Partial matches, abbreviations |
| `TokenSimilarity` | snake_case/camelCase names |
| `CompositeSimilarity` | Weighted combination (default) |

### SchemaWatcher

Continuous monitoring:

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

### Breaking Change Alerts

Enhanced alerting with impact analysis:

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

Format alerts for notifications:

```python
# Slack
slack_payload = alert.format_slack_message()

# Email
subject, html_body = alert.format_email()
```

## Factory Functions

Create pre-configured components:

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

## Integration Examples

### CI/CD Pipeline

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

### Slack Notifications

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

## Best Practices

### 1. Version Strategy Selection

- **Semantic**: Best for APIs and shared schemas
- **Timestamp**: Best for audit trails
- **Git-like**: Best for development environments

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

### 4. Alert Configuration

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

## Related Guides

- [Drift Detection](drift-detection.md) - Data distribution drift
- [Quality Scoring](quality-scoring.md) - Impact on data quality
- [Checkpoint Basics](../checkpoint/basics.md) - CI/CD integration
- [Notifications](../checkpoint/actions/notifications.md) - Alert routing

## API Reference

Full API documentation available at:

- `truthound.profiler.evolution.SchemaEvolutionDetector`
- `truthound.profiler.evolution.SchemaHistory`
- `truthound.profiler.evolution.SchemaWatcher`
- `truthound.profiler.evolution.ColumnRenameDetector`
- `truthound.profiler.evolution.BreakingChangeAlertManager`
- `truthound.profiler.evolution.ImpactAnalyzer`
