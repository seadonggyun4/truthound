# Throttling (Rate Limiting)

A rate limiting system that controls the frequency of notification delivery. It supports multi-level limits (per minute/hour/day) based on the Token Bucket algorithm.

## Overview

```
Notification Request
        │
        ▼
┌───────────────────────────┐
│   NotificationThrottler   │
│                           │
│  1. Check Priority Bypass │
│  2. Get Applicable Limits │
│  3. Acquire Permits       │
│  4. Return ThrottleResult │
└───────────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Allowed] [Throttled]
           retry_after: N seconds
```

## Core Types

### TimeUnit

Time unit enumeration.

```python
from truthound.checkpoint.throttling import TimeUnit

class TimeUnit(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
```

### RateLimitScope

Scope of rate limit application.

```python
from truthound.checkpoint.throttling import RateLimitScope

class RateLimitScope(str, Enum):
    GLOBAL = "global"                    # All notifications combined
    PER_ACTION = "per_action"            # Per action type (slack, email, etc.)
    PER_CHECKPOINT = "per_checkpoint"    # Per checkpoint
    PER_ACTION_CHECKPOINT = "per_action_checkpoint"  # Per action + checkpoint
    PER_SEVERITY = "per_severity"        # Per severity level
    PER_DATA_ASSET = "per_data_asset"    # Per data asset
    CUSTOM = "custom"                    # Custom key
```

### ThrottleStatus

Throttle result status.

```python
from truthound.checkpoint.throttling import ThrottleStatus

class ThrottleStatus(str, Enum):
    ALLOWED = "allowed"          # Request permitted
    THROTTLED = "throttled"      # Rejected due to rate limit exceeded
    QUEUED = "queued"            # Queued for later processing
    BURST_ALLOWED = "burst_allowed"  # Allowed via burst capacity
    ERROR = "error"              # Error occurred
```

### RateLimit

Rate limit configuration.

```python
from truthound.checkpoint.throttling import RateLimit, TimeUnit

@dataclass(frozen=True)
class RateLimit:
    limit: int                  # Maximum request count
    time_unit: TimeUnit         # Time unit
    burst_multiplier: float = 1.0  # Burst capacity multiplier (1.0 = no burst)

# Factory methods
limit = RateLimit.per_minute(10, burst_multiplier=1.5)  # 10/min, burst 15
limit = RateLimit.per_hour(100)                          # 100/hour
limit = RateLimit.per_day(500)                           # 500/day

# Properties
limit.window_seconds      # Window size in seconds
limit.burst_limit         # Burst limit (limit * multiplier)
limit.tokens_per_second   # Token replenishment rate per second
```

### ThrottleResult

Throttle check result.

```python
@dataclass
class ThrottleResult:
    status: ThrottleStatus       # Result status
    key: ThrottlingKey           # Throttling key
    allowed: bool                # Whether allowed
    retry_after: float = 0.0     # Retry wait time in seconds
    remaining: int = 0           # Remaining tokens/requests
    limit: RateLimit | None = None  # Applied rate limit
    message: str = ""            # Message
    metadata: dict = field(default_factory=dict)  # Additional metadata
```

---

## Throttler Types (5 Types)

### TokenBucketThrottler

Token Bucket algorithm. Allows bursts then replenishes tokens at a steady rate.

```python
from truthound.checkpoint.throttling import (
    TokenBucketThrottler,
    RateLimit,
    ThrottlingKey,
    TimeUnit,
)

throttler = TokenBucketThrottler("api")

# Define rate limit
limit = RateLimit.per_minute(10, burst_multiplier=1.5)  # 10/min, burst 15

# Create throttling key
key = ThrottlingKey.for_global(TimeUnit.MINUTE)

# Check only (no token consumption)
result = throttler.check(key, limit)
if result.allowed:
    print(f"Remaining: {result.remaining}")

# Acquire token (check + consume)
result = throttler.acquire(key, limit)
if result.allowed:
    send_notification()
else:
    print(f"Retry after {result.retry_after:.1f}s")
```

**Characteristics:**
- Allows instantaneous burst traffic
- Replenishes tokens at a constant rate
- Smooth rate limiting

### SlidingWindowThrottler

Sliding window algorithm. Provides more accurate rate limiting.

```python
from truthound.checkpoint.throttling import SlidingWindowThrottler

throttler = SlidingWindowThrottler("api")
limit = RateLimit.per_hour(100)
key = ThrottlingKey.for_action("slack", TimeUnit.HOUR)

result = throttler.acquire(key, limit)
```

**Characteristics:**
- No abrupt changes at window boundaries
- More consistent rate limiting
- Slightly higher memory usage

### FixedWindowThrottler

Fixed window algorithm. Simple but allows up to 2x traffic at window boundaries.

```python
from truthound.checkpoint.throttling import FixedWindowThrottler

throttler = FixedWindowThrottler("api")
limit = RateLimit.per_minute(10)
key = ThrottlingKey.for_global(TimeUnit.MINUTE)

result = throttler.acquire(key, limit)
```

**Characteristics:**
- Simple implementation
- Memory efficient
- Can allow up to 2x traffic at window boundaries (edge effect)

### CompositeThrottler

Combines multiple rate limits. Request must pass all limits to be allowed.

```python
from truthound.checkpoint.throttling import CompositeThrottler, RateLimit

throttler = CompositeThrottler("multi-level", algorithm="token_bucket")

# Add multi-level rate limits
throttler.add_limit(RateLimit.per_minute(10))
throttler.add_limit(RateLimit.per_hour(100))
throttler.add_limit(RateLimit.per_day(500))

# Or configure at once
throttler.with_limits([
    RateLimit.per_minute(10),
    RateLimit.per_hour(100),
    RateLimit.per_day(500),
])

key = ThrottlingKey.for_global(TimeUnit.MINUTE)
result = throttler.acquire(key)  # Checks all limits
```

**Characteristics:**
- Multi-level limits (minute/hour/day)
- Must pass all limits
- Most restrictive limit determines final decision

### NoOpThrottler

Pass-through throttler that always allows. For testing or disabling.

```python
from truthound.checkpoint.throttling import NoOpThrottler

throttler = NoOpThrottler()
result = throttler.acquire(key, limit)
assert result.allowed  # Always True
```

---

## NotificationThrottler

High-level throttling service.

```python
from truthound.checkpoint.throttling import (
    NotificationThrottler,
    ThrottlingConfig,
    RateLimitScope,
)

# Create with config
config = ThrottlingConfig(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
    burst_multiplier=1.5,
    scope=RateLimitScope.PER_ACTION,
    algorithm="token_bucket",
    enabled=True,
    priority_bypass=True,         # Bypass for critical notifications
    priority_threshold="critical",  # Critical bypasses limit
)

throttler = NotificationThrottler(config=config)

# Check
result = throttler.check(
    action_type="slack",
    checkpoint_name="data_quality",
    severity="high",
)

# Acquire
result = throttler.acquire(
    action_type="slack",
    checkpoint_name="data_quality",
    severity="high",
)

if result.allowed:
    send_notification()

# Check with CheckpointResult
result = throttler.acquire_result(
    checkpoint_result=checkpoint_result,
    action_type="slack",
)

# Convenience method
if throttler.is_throttled("slack", "my_checkpoint"):
    print("Rate limit exceeded")
```

### ThrottlingConfig

```python
@dataclass
class ThrottlingConfig:
    per_minute_limit: int | None = 10    # Per-minute limit
    per_hour_limit: int | None = 100     # Per-hour limit
    per_day_limit: int | None = 500      # Per-day limit
    burst_multiplier: float = 1.0        # Burst multiplier
    scope: RateLimitScope = RateLimitScope.GLOBAL  # Scope
    algorithm: str = "token_bucket"      # Algorithm
    enabled: bool = True                 # Enabled
    custom_limits: dict[str, list[RateLimit]] = {}  # Per-action custom limits
    severity_limits: dict[str, list[RateLimit]] = {}  # Per-severity limits
    priority_bypass: bool = False        # Priority bypass
    priority_threshold: str = "critical" # Bypass threshold
    queue_on_throttle: bool = False      # Queue when throttled
    max_queue_size: int = 1000           # Maximum queue size
```

---

## ThrottlerBuilder

Configure throttler with fluent API.

```python
from truthound.checkpoint.throttling import ThrottlerBuilder, RateLimitScope

throttler = (
    ThrottlerBuilder()
    # Default limits
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .with_per_day_limit(500)
    # Burst configuration
    .with_burst_allowance(1.5)
    # Algorithm and scope
    .with_algorithm("token_bucket")  # sliding_window, fixed_window
    .with_scope(RateLimitScope.PER_ACTION)
    # Priority bypass (critical notifications bypass)
    .with_priority_bypass("critical")
    # Per-action custom limits
    .with_action_limit("pagerduty", per_minute=5, per_hour=20)
    .with_action_limit("slack", per_minute=20, per_hour=200)
    # Per-severity limits
    .with_severity_limit("info", per_minute=5, per_hour=50)
    .with_severity_limit("critical", per_minute=None, per_hour=None)  # No limit
    # Enable queueing
    .with_queueing(max_size=1000)
    .build()
)
```

---

## Middleware

Automatically wraps actions to apply throttling.

### ThrottlingMiddleware

```python
from truthound.checkpoint.throttling import (
    ThrottlingMiddleware,
    ThrottlerBuilder,
)
from truthound.checkpoint.actions import SlackNotification

# Create throttler
throttler = (
    ThrottlerBuilder()
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .build()
)

# Create middleware
middleware = ThrottlingMiddleware(throttler=throttler)

# Wrap action
slack_action = SlackNotification(webhook_url="...")
throttled_action = middleware.wrap(slack_action)

# Use - throttling automatically applied
await throttled_action.execute(checkpoint_result)
```

### @throttled Decorator

```python
from truthound.checkpoint.throttling import throttled, RateLimit

@throttled(
    limits=[
        RateLimit.per_minute(10),
        RateLimit.per_hour(100),
    ]
)
async def send_notification(result):
    # Notification logic
    pass
```

### Global Throttling

```python
from truthound.checkpoint.throttling import (
    configure_global_throttling,
    get_global_middleware,
)

# Global configuration
configure_global_throttling(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
)

# Use global middleware
middleware = get_global_middleware()
throttled_action = middleware.wrap(slack_action)
```

---

## Storage Backend

### InMemoryThrottlingStore

In-memory store for single-process use.

```python
from truthound.checkpoint.throttling import InMemoryThrottlingStore

store = InMemoryThrottlingStore()

# Used internally by NotificationThrottler
```

---

## Statistics Retrieval

```python
stats = throttler.get_stats()

print(f"Total checked: {stats.total_checked}")
print(f"Total allowed: {stats.total_allowed}")
print(f"Total throttled: {stats.total_throttled}")
print(f"Burst allowed: {stats.total_burst_allowed}")
print(f"Throttle rate: {stats.throttle_rate:.2%}")
print(f"Allow rate: {stats.allow_rate:.2%}")
print(f"Active buckets: {stats.buckets_active}")
```

---

## Complete Example

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.throttling import (
    ThrottlerBuilder,
    ThrottlingMiddleware,
    RateLimitScope,
)
from truthound.checkpoint.actions import SlackNotification, PagerDutyAction

# Multi-level throttler
throttler = (
    ThrottlerBuilder()
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .with_per_day_limit(500)
    .with_burst_allowance(1.5)
    .with_scope(RateLimitScope.PER_ACTION)
    .with_priority_bypass("critical")
    .with_action_limit("pagerduty", per_minute=5, per_hour=20)
    .with_action_limit("slack", per_minute=20, per_hour=200)
    .build()
)

middleware = ThrottlingMiddleware(throttler=throttler)

# Wrap actions
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
throttling:
  enabled: true
  per_minute_limit: 10
  per_hour_limit: 100
  per_day_limit: 500
  burst_multiplier: 1.5
  scope: per_action
  algorithm: token_bucket
  priority_bypass: true
  priority_threshold: critical
  action_limits:
    pagerduty:
      per_minute: 5
      per_hour: 20
    slack:
      per_minute: 20
      per_hour: 200
  severity_limits:
    info:
      per_minute: 5
    critical:
      per_minute: null  # No limit
```

---

## Throttler Comparison

| Throttler | Use Case | Burst | Accuracy | Memory |
|-----------|----------|-------|----------|--------|
| **TokenBucket** | General | Allowed | Medium | Low |
| **SlidingWindow** | Precise limiting | None | High | Medium |
| **FixedWindow** | Simple limiting | 2x possible | Low | Low |
| **Composite** | Multi-level limits | Per config | High | Medium |
| **NoOp** | Testing | N/A | N/A | None |
