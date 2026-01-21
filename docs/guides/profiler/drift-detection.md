# Drift Detection

This document describes the schema change and data drift detection system.

## Overview

The drift detection system implemented in `src/truthound/profiler/evolution/detector.py` tracks schema and data changes over time.

## SchemaChangeType

```python
class SchemaChangeType(str, Enum):
    """Schema change types"""

    COLUMN_ADDED = "column_added"       # New column added
    COLUMN_REMOVED = "column_removed"   # Column removed
    COLUMN_RENAMED = "column_renamed"   # Column renamed
    TYPE_CHANGED = "type_changed"       # Data type changed
```

## SchemaChange

```python
@dataclass
class SchemaChange:
    """Schema change information"""

    change_type: SchemaChangeType
    column_name: str
    old_value: Any = None   # Previous value (type, name, etc.)
    new_value: Any = None   # New value
    severity: str = "medium"
    description: str = ""
```

## SchemaChangeDetector Protocol

```python
from typing import Protocol

class SchemaChangeDetector(Protocol):
    """Schema change detector protocol"""

    def detect_changes(
        self,
        old_profile: TableProfile,
        new_profile: TableProfile,
    ) -> list[SchemaChange]:
        """Detect schema changes between two profiles"""
        ...
```

## Type Compatibility Mapping

Defines safe type upgrades.

```python
# Compatible type conversions (safe upgrade)
TYPE_COMPATIBILITY = {
    "Int8": ["Int16", "Int32", "Int64", "Float32", "Float64"],
    "Int16": ["Int32", "Int64", "Float32", "Float64"],
    "Int32": ["Int64", "Float64"],
    "Int64": ["Float64"],
    "Float32": ["Float64"],
    "Utf8": ["LargeUtf8"],
}

def is_compatible_change(old_type: str, new_type: str) -> bool:
    """Check if type change is compatible"""
    return new_type in TYPE_COMPATIBILITY.get(old_type, [])
```

## Basic Usage

```python
from truthound.profiler.evolution import SchemaEvolutionDetector

detector = SchemaEvolutionDetector()

# Detect schema changes
changes = detector.detect_changes(old_profile, new_profile)

for change in changes:
    print(f"Type: {change.change_type}")
    print(f"Column: {change.column_name}")
    print(f"Severity: {change.severity}")
    if change.change_type == SchemaChangeType.TYPE_CHANGED:
        print(f"  {change.old_value} -> {change.new_value}")
```

## Column Rename Detection

Infers renames by analyzing columns with similar statistics.

```python
from truthound.profiler.evolution import ColumnRenameDetector

detector = ColumnRenameDetector(
    similarity_threshold=0.9,  # 90% or higher similarity
)

renames = detector.detect_renames(old_profile, new_profile)

for rename in renames:
    print(f"Rename detected: {rename.old_name} -> {rename.new_name}")
    print(f"Confidence: {rename.confidence:.2%}")
```

## Compatibility Analysis

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

## Drift Severity Levels

| Severity | Description | Example |
|----------|-------------|---------|
| `info` | Informational change | New column added |
| `low` | Minor change | Compatible type expansion |
| `medium` | Attention required | Column renamed |
| `high` | Investigation needed | Incompatible type change |
| `critical` | Immediate action required | Required column removed |

## Breaking Change Alerts

```python
from truthound.profiler.evolution import BreakingChangeAlert

alerts = detector.get_breaking_alerts(changes)

for alert in alerts:
    print(f"ALERT: {alert.message}")
    print(f"Impact: {alert.impact}")
    print(f"Recommendation: {alert.recommendation}")
```

## History Tracking

```python
from truthound.profiler.evolution import SchemaHistory

history = SchemaHistory(storage_dir=".truthound/schema_history")

# Save profile
history.save(profile, version="v1.0")
history.save(new_profile, version="v1.1")

# Retrieve history
versions = history.list_versions()

# Compare versions
changes = history.compare("v1.0", "v1.1")

# Load specific version
old_profile = history.load("v1.0")
```

## Automatic Alerting

```python
from truthound.profiler.evolution import SchemaWatcher

watcher = SchemaWatcher(
    alert_callback=lambda alert: send_slack_notification(alert),
    check_interval_minutes=60,
)

# Start monitoring
watcher.watch("data.csv", baseline_profile)

# Automatic alert sent when changes are detected
```

## CLI Usage

```bash
# Compare two profiles
th compare profile_v1.json profile_v2.json

# Detect schema changes
th schema-diff old_profile.json new_profile.json

# Compatibility analysis
th check-compatibility old_profile.json new_profile.json

# Check for breaking changes
th check-breaking old_profile.json new_profile.json
```

## Integration Example

```python
from truthound.profiler import TableProfiler
from truthound.profiler.evolution import SchemaEvolutionDetector
from truthound.profiler.caching import ProfileCache

# Set up profiler and cache
profiler = TableProfiler()
cache = ProfileCache()
detector = SchemaEvolutionDetector()

# Baseline profile (load from cache or create)
baseline_key = cache.compute_fingerprint("data_baseline.csv")
baseline = cache.get_or_compute(
    baseline_key,
    lambda: profiler.profile_file("data_baseline.csv"),
)

# Current profile
current = profiler.profile_file("data_current.csv")

# Detect changes
changes = detector.detect_changes(baseline, current)

if changes:
    print(f"Found {len(changes)} schema changes:")
    for change in changes:
        print(f"  - {change.change_type}: {change.column_name}")

    # Check for breaking changes
    breaking = [c for c in changes if c.severity == "critical"]
    if breaking:
        raise ValueError(f"Breaking changes detected: {breaking}")
```

## Next Steps

- [Quality Scoring](quality-scoring.md) - Impact of drift on quality
- [Visualization](visualization.md) - Generate drift reports
