# Triggers

Triggers determine the automatic execution timing of Checkpoints. They support time-based, cron expression, event-based, and file watch methods.

## Trigger Base Class

All triggers inherit from `BaseTrigger[ConfigT]`.

```python
# src/truthound/checkpoint/triggers/base.py

class TriggerStatus(str, Enum):
    """Trigger status."""
    ACTIVE = "active"      # Active
    PAUSED = "paused"      # Paused
    STOPPED = "stopped"    # Stopped
    ERROR = "error"        # Error


@dataclass
class TriggerConfig:
    """Trigger base configuration."""
    name: str | None = None           # Trigger name
    enabled: bool = True              # Enable status
    max_runs: int = 0                 # Maximum run count (0 = unlimited)
    run_immediately: bool = False     # Execute immediately on start
    catch_up: bool = False            # Catch up missed executions
    max_concurrent: int = 1           # Maximum concurrent executions
    metadata: dict = field(default_factory=dict)


@dataclass
class TriggerResult:
    """Trigger evaluation result."""
    should_run: bool          # Whether to run
    reason: str = ""          # Reason
    next_run: datetime | None = None  # Next run time
    context: dict = field(default_factory=dict)  # Additional context
```

---

## ScheduleTrigger

Time interval-based trigger.

### Configuration (ScheduleConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `interval_seconds` | `int` | `0` | Interval (seconds) |
| `interval_minutes` | `int` | `0` | Interval (minutes) |
| `interval_hours` | `int` | `0` | Interval (hours) |
| `start_time` | `datetime \| None` | `None` | Start time (None = immediately) |
| `end_time` | `datetime \| None` | `None` | End time (None = unlimited) |
| `run_on_weekdays` | `list[int] \| None` | `None` | Days to run (0=Mon, 6=Sun) |
| `timezone` | `str \| None` | `None` | Timezone |

### Usage Examples

```python
from truthound.checkpoint.triggers import ScheduleTrigger
from datetime import datetime

# Run every hour
trigger = ScheduleTrigger(interval_hours=1)

# Run every 30 minutes
trigger = ScheduleTrigger(interval_minutes=30)

# Run on weekdays (Mon-Fri) during business hours only
trigger = ScheduleTrigger(
    interval_minutes=30,
    run_on_weekdays=[0, 1, 2, 3, 4],  # Mon-Fri
    start_time=datetime(2024, 1, 1, 9, 0),   # From 9 AM
    end_time=datetime(2024, 12, 31, 18, 0),  # Until 6 PM
)

# Specify timezone
trigger = ScheduleTrigger(
    interval_hours=1,
    timezone="America/New_York",
)
```

### Connect to Checkpoint

```python
from truthound.checkpoint import Checkpoint

checkpoint = Checkpoint(
    name="hourly_check",
    data_source="data.csv",
    validators=["null"],
)

# Add trigger
checkpoint.add_trigger(ScheduleTrigger(interval_hours=1))
```

---

## CronTrigger

Trigger using standard cron expressions.

### Configuration (CronConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `expression` | `str` | `""` | Cron expression (5 or 6 fields) |
| `timezone` | `str \| None` | `None` | Timezone |

### Cron Expression Format

**5 fields** (standard):
```
minute hour day month weekday
```

**6 fields** (with seconds):
```
second minute hour day month weekday
```

### Field Values

| Field | Range | Special Characters |
|-------|-------|-------------------|
| Second (optional) | 0-59 | `*`, `,`, `-`, `/` |
| Minute | 0-59 | `*`, `,`, `-`, `/` |
| Hour | 0-23 | `*`, `,`, `-`, `/` |
| Day | 1-31 | `*`, `,`, `-`, `/` |
| Month | 1-12 | `*`, `,`, `-`, `/` |
| Weekday | 0-6 (0=Sun) | `*`, `,`, `-`, `/` |

### Usage Examples

```python
from truthound.checkpoint.triggers import CronTrigger

# Every day at midnight
trigger = CronTrigger(expression="0 0 * * *")

# Every hour on the hour
trigger = CronTrigger(expression="0 * * * *")

# Every 15 minutes
trigger = CronTrigger(expression="*/15 * * * *")

# Every Monday at 9 AM
trigger = CronTrigger(expression="0 9 * * 1")

# Weekdays at 9 AM
trigger = CronTrigger(expression="0 9 * * 1-5")

# First day of every month at midnight
trigger = CronTrigger(expression="0 0 1 * *")

# 6 fields: Every day at 9:00:30
trigger = CronTrigger(expression="30 0 9 * * *")

# Specify timezone
trigger = CronTrigger(
    expression="0 9 * * *",
    timezone="Asia/Seoul",
)
```

---

## EventTrigger

Triggered by external events.

### Configuration (EventConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `event_type` | `str` | `""` | Event type (for filtering) |
| `event_filter` | `dict` | `{}` | Event filter conditions |
| `debounce_seconds` | `int` | `0` | Debounce time (seconds) |
| `batch_events` | `bool` | `False` | Batch event processing |
| `batch_window_seconds` | `int` | `30` | Batch window (seconds) |

### Usage Examples

```python
from truthound.checkpoint.triggers import EventTrigger

# Basic event trigger
trigger = EventTrigger(event_type="data_updated")

# Add filter conditions
trigger = EventTrigger(
    event_type="data_updated",
    event_filter={
        "source": "production",
        "priority": "high",
    },
)

# Debounce: Ignore duplicate events within 60 seconds
trigger = EventTrigger(
    event_type="data_updated",
    debounce_seconds=60,
)

# Batch processing: Collect events for 30 seconds then process at once
trigger = EventTrigger(
    event_type="data_updated",
    batch_events=True,
    batch_window_seconds=30,
)
```

### Fire Events

```python
# Programmatically fire events
trigger.fire_event({
    "source": "production",
    "priority": "high",
    "table": "users",
    "rows_affected": 1500,
})
```

---

## FileWatchTrigger

Triggers on file system changes.

### Configuration (FileWatchConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `paths` | `list[str]` | `[]` | Paths to watch |
| `patterns` | `list[str]` | `["*"]` | File patterns (glob) |
| `recursive` | `bool` | `True` | Include subdirectories |
| `events` | `list[str]` | `["modified"]` | Events to detect |
| `ignore_patterns` | `list[str]` | `[]` | Patterns to ignore |
| `hash_check` | `bool` | `True` | Hash-based change detection |
| `poll_interval_seconds` | `int` | `5` | Polling interval (seconds) |

### Event Types

| Event | Description |
|-------|-------------|
| `modified` | File modified |
| `created` | File created |
| `deleted` | File deleted |

### Usage Examples

```python
from truthound.checkpoint.triggers import FileWatchTrigger

# Watch specific directory
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv", "*.parquet"],
)

# Multiple paths and recursive watch
trigger = FileWatchTrigger(
    paths=["./data", "/shared/datasets"],
    patterns=["*.csv", "*.parquet", "*.json"],
    recursive=True,
    events=["modified", "created"],
)

# Exclude specific files
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv"],
    ignore_patterns=[".*", "__pycache__", "*.tmp", "test_*"],
)

# Hash-based change detection (triggers only on actual content changes)
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv"],
    hash_check=True,  # Won't trigger on timestamp changes alone
    poll_interval_seconds=10,
)
```

---

## Usage with CheckpointRunner

```python
from truthound.checkpoint import Checkpoint, CheckpointRunner
from truthound.checkpoint.triggers import ScheduleTrigger, CronTrigger

# Create checkpoint with triggers
hourly_check = Checkpoint(
    name="hourly_metrics",
    data_source="metrics.csv",
    validators=["null"],
)
hourly_check.add_trigger(ScheduleTrigger(interval_hours=1))

daily_check = Checkpoint(
    name="daily_data_validation",
    data_source="data.parquet",
    validators=["range", "distribution"],
)
daily_check.add_trigger(CronTrigger(expression="0 0 * * *"))

# Create and run the runner
runner = CheckpointRunner(
    max_workers=4,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
)

runner.add_checkpoint(hourly_check)
runner.add_checkpoint(daily_check)

# Start background execution (trigger monitoring)
runner.start()

# ... application logic ...

# Shutdown
runner.stop()
```

---

## YAML Configuration Example

```yaml
checkpoints:
  - name: hourly_metrics
    data_source: metrics.csv
    validators:
      - "null"
    triggers:
      # Every hour
      - type: schedule
        interval_hours: 1
        run_immediately: true

  - name: daily_data_validation
    data_source: data.parquet
    validators:
      - range
      - distribution
    triggers:
      # Every day at midnight
      - type: cron
        expression: "0 0 * * *"
        timezone: Asia/Seoul

  - name: file_based_check
    data_source: ./data/users.csv
    validators:
      - "null"
    triggers:
      # On file change
      - type: file_watch
        paths:
          - ./data
        patterns:
          - "*.csv"
        events:
          - modified
          - created
        hash_check: true
```

---

## Trigger Comparison

| Trigger | Use Case | Advantages | Disadvantages |
|---------|----------|------------|---------------|
| **ScheduleTrigger** | Regular execution | Simple, predictable | Difficult to specify exact times |
| **CronTrigger** | Complex schedules | Flexible expressions | Learning curve required |
| **EventTrigger** | Event-based | Immediate response | Requires external system |
| **FileWatchTrigger** | File changes | Automation | Resource consumption |
