# Notification Deduplication

A system for preventing duplicate notifications. It suppresses repeated delivery of identical notifications based on time windows.

## Overview

```
Notification Request
        │
        ▼
┌───────────────────────────┐
│ NotificationDeduplicator  │
│                           │
│  1. Generate Fingerprint  │
│  2. Check Window Store    │
│  3. Decide: Allow/Suppress│
└───────────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Allow]   [Suppress]
```

## Core Classes

### TimeWindow

A time window for deduplication.

```python
from truthound.checkpoint.deduplication import TimeWindow, WindowUnit

class WindowUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

# 5-minute window
window = TimeWindow(seconds=300)
window = TimeWindow(minutes=5)

# 1-hour window
window = TimeWindow(hours=1)

# 1-day window
window = TimeWindow(days=1)

# Combined
window = TimeWindow(hours=1, minutes=30)  # 90 minutes
```

### NotificationFingerprint

A unique identifier for notifications.

```python
@dataclass
class NotificationFingerprint:
    """Notification fingerprint."""
    key: str                  # Hash key (unique identifier)
    checkpoint_name: str      # Checkpoint name
    action_type: str          # Action type (slack, email, etc.)
    components: dict[str, Any] = field(default_factory=dict)  # Components used for fingerprint generation
    created_at: datetime = field(default_factory=datetime.now)  # Creation time
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    @classmethod
    def generate(
        cls,
        checkpoint_name: str,
        action_type: str,
        *,
        severity: str | None = None,      # Severity (optional)
        data_asset: str | None = None,    # Data asset (optional)
        issue_types: Sequence[str] | None = None,  # Issue types (optional)
        custom_key: str | None = None,    # Custom key (optional)
        algorithm: str = "sha256",        # Hash algorithm
        **extra_components: Any,
    ) -> "NotificationFingerprint": ...
```

### DeduplicationPolicy

Deduplication policies.

```python
from truthound.checkpoint.deduplication import DeduplicationPolicy

class DeduplicationPolicy(str, Enum):
    NONE = "none"           # No deduplication
    BASIC = "basic"         # Differentiate by checkpoint + action_type
    SEVERITY = "severity"   # + differentiate by severity
    ISSUE_BASED = "issue_based"  # + differentiate by issue types
    STRICT = "strict"       # Use full fingerprint
    CUSTOM = "custom"       # User-defined function
```

---

## Basic Usage

```python
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    DeduplicationConfig,
    InMemoryDeduplicationStore,
    TimeWindow,
    DeduplicationPolicy,
)

# Deduplication configuration
config = DeduplicationConfig(
    policy=DeduplicationPolicy.SEVERITY,
    default_window=TimeWindow(minutes=5),
)

# Create deduplicator
deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    config=config,
)

# Check for duplicates (using checkpoint_result and action_type)
result = deduplicator.check(checkpoint_result, "slack", severity="high")

if result.should_send:
    # Send notification
    await action.execute(checkpoint_result)
    # Record delivery
    deduplicator.mark_sent(result.fingerprint)
else:
    print(f"Notification suppressed: {result.message}")

# Or simply use is_duplicate
if not deduplicator.is_duplicate(checkpoint_result, "slack", severity="high"):
    await action.execute(checkpoint_result)
```

---

## Store Backends

### InMemoryDeduplicationStore

In-memory store for single-process use.

```python
from truthound.checkpoint.deduplication import InMemoryDeduplicationStore

store = InMemoryDeduplicationStore(
    max_size=10000,  # Maximum record count
    auto_cleanup_interval=60,  # Auto cleanup interval (seconds)
)

deduplicator = NotificationDeduplicator(store=store)
```

### RedisStreamsDeduplicationStore

Redis store for distributed environments.

```python
from truthound.checkpoint.deduplication import RedisStreamsDeduplicationStore

store = RedisStreamsDeduplicationStore(
    redis_url="redis://localhost:6379",
    stream_key="truthound:dedup:stream",
    max_stream_length=10000,
)

deduplicator = NotificationDeduplicator(store=store)
```

---

## Window Strategies (4 Types)

### SlidingWindowStrategy

A fixed time window. Suppresses all notifications within the window.

```python
from truthound.checkpoint.deduplication import SlidingWindowStrategy, TimeWindow

strategy = SlidingWindowStrategy(
    window=TimeWindow(minutes=5),
)

# Example:
# 10:00:00 - First notification → Allowed
# 10:02:00 - Same notification → Suppressed (within 5 minutes)
# 10:06:00 - Same notification → Allowed (5 minutes elapsed)
```

### TumblingWindowStrategy

Non-overlapping fixed buckets.

```python
from truthound.checkpoint.deduplication import TumblingWindowStrategy

strategy = TumblingWindowStrategy(
    bucket_size=TimeWindow(minutes=15),
)

# Example: 15-minute buckets
# 10:00-10:15 bucket: Only first notification allowed
# 10:15-10:30 bucket: New bucket, first notification allowed
```

### SessionWindowStrategy

Event-based sessions. Session ends after a period of no notifications.

```python
from truthound.checkpoint.deduplication import SessionWindowStrategy

strategy = SessionWindowStrategy(
    gap=TimeWindow(minutes=10),  # New session if no notifications for 10 minutes
)

# Example:
# 10:00 - Notification → Session start, allowed
# 10:05 - Notification → Within session, suppressed
# 10:20 - Notification → Gap exceeded 10 minutes, new session, allowed
```

### AdaptiveWindowStrategy (Planned)

Dynamically adjusts window size based on notification frequency.

---

## Per-Action Window Configuration

Different windows can be configured for each action type.

```python
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    DeduplicationConfig,
    TimeWindow,
)

config = DeduplicationConfig(
    enabled=True,
    policy=DeduplicationPolicy.SEVERITY,
    default_window=TimeWindow(minutes=5),
    # Per-action windows
    action_windows={
        "pagerduty": TimeWindow(hours=1),     # PagerDuty: 1 hour
        "slack": TimeWindow(minutes=5),        # Slack: 5 minutes
        "email": TimeWindow(hours=24),         # Email: 24 hours
    },
    # Per-severity windows
    severity_windows={
        "critical": TimeWindow(minutes=1),     # Critical: 1 minute
        "high": TimeWindow(minutes=5),         # High: 5 minutes
        "medium": TimeWindow(minutes=15),      # Medium: 15 minutes
    },
)

deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    config=config,
)
```

---

## Middleware Usage

Automatically wraps actions to apply deduplication.

```python
from truthound.checkpoint.deduplication import (
    DeduplicationMiddleware,
    deduplicated,
)
from truthound.checkpoint.actions import SlackNotification

# Using middleware
middleware = DeduplicationMiddleware(
    deduplicator=deduplicator,
)

slack_action = SlackNotification(webhook_url="...")
deduplicated_action = middleware.wrap(slack_action)

# Or using decorator
@deduplicated(window=TimeWindow(minutes=5))
async def send_notification(result):
    # Notification logic
    pass
```

---

## Statistics Retrieval

```python
# Deduplication statistics
stats = deduplicator.get_stats()

print(f"Total evaluated: {stats.total_evaluated}")
print(f"Total suppressed: {stats.suppressed}")
print(f"Suppression ratio: {stats.suppression_ratio:.2%}")
print(f"Active fingerprints: {stats.active_fingerprints}")
```

---

## Complete Example

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    InMemoryDeduplicationStore,
    DeduplicationConfig,
    DeduplicationPolicy,
    DeduplicationMiddleware,
    TimeWindow,
)
from truthound.checkpoint.actions import SlackNotification, PagerDutyAction

# Deduplication configuration
config = DeduplicationConfig(
    enabled=True,
    policy=DeduplicationPolicy.SEVERITY,
    default_window=TimeWindow(minutes=5),
    action_windows={
        "pagerduty": TimeWindow(hours=1),
        "slack": TimeWindow(minutes=5),
    },
    severity_windows={
        "critical": TimeWindow(minutes=1),
    },
)

deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    config=config,
)

# Wrap actions with middleware
middleware = DeduplicationMiddleware(deduplicator=deduplicator)

slack_action = middleware.wrap(
    SlackNotification(webhook_url="${SLACK_WEBHOOK}")
)

pagerduty_action = middleware.wrap(
    PagerDutyAction(routing_key="${PAGERDUTY_KEY}")
)

# Apply to checkpoint
checkpoint = Checkpoint(
    name="production_check",
    data_source="data.csv",
    validators=["null"],
    actions=[slack_action, pagerduty_action],
)
```

---

## YAML Configuration

```yaml
deduplication:
  enabled: true
  policy: severity
  default_window:
    minutes: 5
  action_windows:
    pagerduty:
      hours: 1
    slack:
      minutes: 5
    email:
      hours: 24
  severity_windows:
    critical:
      minutes: 1
    high:
      minutes: 5
  store:
    type: redis  # or memory
    redis_url: redis://localhost:6379
```
