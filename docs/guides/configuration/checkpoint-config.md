# Checkpoint Configuration

Truthound checkpoints orchestrate validation pipelines with actions, routing, and notifications.

## Quick Start

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig

config = CheckpointConfig(
    name="daily_data_validation",
    data_source="users_table",
    validators=["not_null", "unique"],
    fail_on_critical=True,
)

checkpoint = Checkpoint(config)
result = checkpoint.run()
```

## CheckpointConfig

```python
from truthound.checkpoint import CheckpointConfig

config = CheckpointConfig(
    name="my_checkpoint",
    data_source="",                     # Data source identifier
    validators=None,                    # Validators to run
    validator_config={},                # Validator-specific config
    min_severity=None,                  # Minimum severity to report
    schema=None,                        # Schema path or object
    auto_schema=False,                  # Auto-generate schema
    run_name_template="%Y%m%d_%H%M%S",  # Run ID template
    tags={},                            # Custom tags
    metadata={},                        # Custom metadata
    fail_on_critical=True,              # Fail on critical issues
    fail_on_high=False,                 # Fail on high severity
    timeout_seconds=3600,               # Timeout (1 hour)
    sample_size=None,                   # Sample size (None = full)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | default_checkpoint | Checkpoint name |
| `fail_on_critical` | True | Fail on critical issues |
| `fail_on_high` | False | Fail on high severity issues |
| `timeout_seconds` | 3600 | Execution timeout |
| `auto_schema` | False | Auto-generate schema |

## Action Configuration

### ActionConfig Base

```python
from truthound.checkpoint.actions.base import ActionConfig, NotifyCondition

config = ActionConfig(
    name=None,                          # Action name
    enabled=True,                       # Enable/disable
    notify_on=NotifyCondition.ALWAYS,   # When to notify
    timeout_seconds=30,                 # Action timeout
    retry_count=0,                      # Retry attempts
    retry_delay_seconds=1.0,            # Retry delay
    fail_checkpoint_on_error=False,     # Fail checkpoint on error
    metadata={},                        # Custom metadata
)
```

**Notify Conditions:**

| Condition | Description |
|-----------|-------------|
| `ALWAYS` | Always execute |
| `SUCCESS` | Only on success |
| `FAILURE` | Only on failure |
| `ERROR` | Only on error |
| `WARNING` | Only on warning |
| `FAILURE_OR_ERROR` | On failure or error |
| `NOT_SUCCESS` | On anything but success |

### Slack Notification

```python
from truthound.checkpoint.actions.slack_notify import (
    SlackNotification,
    SlackConfig,
)

config = SlackConfig(
    webhook_url="https://hooks.slack.com/...",
    channel="#data-quality",            # Override channel
    username="Truthound",
    icon_emoji=":mag:",
    include_details=True,               # Include issue details
    mention_on_failure=["@oncall"],     # Users to mention
    custom_message=None,                # Custom message template
    notify_on=NotifyCondition.FAILURE,
)

action = SlackNotification(config)
```

### Email Notification

```python
from truthound.checkpoint.actions.email_notify import (
    EmailNotification,
    EmailConfig,
)

config = EmailConfig(
    # SMTP settings
    smtp_host="smtp.example.com",
    smtp_port=587,
    smtp_user="user@example.com",
    smtp_password="secret",
    use_tls=True,
    use_ssl=False,

    # Email settings
    from_address="truthound@example.com",
    to_addresses=["team@example.com"],
    cc_addresses=[],
    subject_template="[Truthound] {status} - {checkpoint}",
    include_html=True,

    # Provider (smtp, sendgrid, ses)
    provider="smtp",
    api_key=None,                       # For SendGrid/SES

    notify_on=NotifyCondition.FAILURE,
)

action = EmailNotification(config)
```

### PagerDuty

```python
from truthound.checkpoint.actions.pagerduty import (
    PagerDutyAction,
    PagerDutyConfig,
)

config = PagerDutyConfig(
    routing_key="your-routing-key",
    severity="error",                   # critical, error, warning, info
    auto_severity=True,                 # Map from validation severity
    component="data-quality",
    group="truthound",
    class_type="validation",
    custom_details={},
    dedup_key_template="{checkpoint}_{data_asset}",
    resolve_on_success=True,            # Auto-resolve on success
    api_endpoint="https://events.pagerduty.com/v2/enqueue",
    notify_on=NotifyCondition.FAILURE_OR_ERROR,
)

action = PagerDutyAction(config)
```

### Webhook

```python
from truthound.checkpoint.actions.webhook import (
    WebhookAction,
    WebhookConfig,
)

config = WebhookConfig(
    url="https://api.example.com/webhook",
    method="POST",
    headers={"Content-Type": "application/json"},

    # Authentication
    auth_type="bearer",                 # none, basic, bearer, api_key
    auth_credentials={"token": "xxx"},

    payload_template=None,              # Custom payload template
    include_full_result=True,
    ssl_verify=True,
    success_codes=[200, 201, 202, 204],
    notify_on=NotifyCondition.ALWAYS,
)

action = WebhookAction(config)
```

### Store Result

```python
from truthound.checkpoint.actions.store_result import (
    StoreResultAction,
    StoreResultConfig,
)

config = StoreResultConfig(
    store_path="./truthound_results",
    store_type="file",                  # file, s3, gcs, database
    format="json",                      # json, yaml, parquet
    partition_by="date",                # date, checkpoint, status
    retention_days=0,                   # 0 = unlimited
    include_validation_details=True,
    compress=False,
    notify_on=NotifyCondition.ALWAYS,
)

action = StoreResultAction(config)
```

## Routing Configuration

### ActionRouter

```python
from truthound.checkpoint.routing.base import (
    ActionRouter,
    Route,
    RouteMode,
    RoutePriority,
)

router = ActionRouter(
    mode=RouteMode.ALL_MATCHES,         # Routing mode
    default_actions=[],                 # Fallback actions
)
```

**Route Modes:**

| Mode | Description |
|------|-------------|
| `FIRST_MATCH` | Stop at first matching route |
| `ALL_MATCHES` | Execute all matching routes |
| `PRIORITY_GROUP` | Execute highest priority group |

**Route Priority:**

| Priority | Value |
|----------|-------|
| `CRITICAL` | 100 |
| `HIGH` | 80 |
| `NORMAL` | 50 |
| `LOW` | 20 |
| `DEFAULT` | 0 |

### Route Configuration

```python
from truthound.checkpoint.routing.rules import (
    SeverityRule,
    IssueCountRule,
    StatusRule,
    PassRateRule,
    AllOf,
    AnyOf,
)

# Create routes
route = Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[pagerduty_action],
    priority=RoutePriority.CRITICAL,
    enabled=True,
    stop_on_match=False,
)

router.add_route(route)

# Complex rule with combinators
complex_rule = AllOf([
    SeverityRule(min_severity="high"),
    IssueCountRule(min_issues=10),
])

router.add_route(Route(
    name="high_volume_alerts",
    rule=complex_rule,
    actions=[slack_action, email_action],
))
```

### Available Rules

| Rule | Description |
|------|-------------|
| `SeverityRule` | Match by minimum severity |
| `IssueCountRule` | Match by issue count |
| `StatusRule` | Match by status |
| `TagRule` | Match by tags |
| `PassRateRule` | Match by pass rate |
| `TimeWindowRule` | Match by time |
| `DataAssetRule` | Match by data asset |
| `MetadataRule` | Match by metadata |
| `ErrorRule` | Match on error |
| `AlwaysRule` | Always match |
| `NeverRule` | Never match |

**Combinators:**

| Combinator | Description |
|------------|-------------|
| `AllOf` | All rules must match (AND) |
| `AnyOf` | Any rule must match (OR) |
| `NotRule` | Negate a rule |

### YAML Configuration

```yaml
# routing.yaml
mode: all_matches

default_actions:
  - type: slack
    webhook_url: "${SLACK_WEBHOOK}"
    notify_on: failure

routes:
  - name: critical_alerts
    rule:
      type: severity
      min_severity: critical
    actions:
      - type: pagerduty
        routing_key: "${PAGERDUTY_KEY}"
    priority: critical
    enabled: true

  - name: high_volume
    rule:
      type: all_of
      rules:
        - type: severity
          min_severity: high
        - type: issue_count
          min_count: 10
    actions:
      - type: email
        to_addresses: ["team@example.com"]
```

```python
from truthound.checkpoint.routing.config import RouteConfigParser

parser = RouteConfigParser()
router = parser.parse_file("routing.yaml")
```

## Deduplication Configuration

Prevent duplicate notifications.

```python
from truthound.checkpoint.deduplication.protocols import (
    TimeWindow,
    WindowUnit,
    NotificationFingerprint,
)

# Time window configuration
window = TimeWindow(
    value=300,
    unit=WindowUnit.SECONDS,
)

# Convenience constructors
window = TimeWindow(minutes=5)
window = TimeWindow(hours=1)
window = TimeWindow(days=1)

# Generate fingerprint
fingerprint = NotificationFingerprint.generate(
    checkpoint_name="daily_data_validation",
    action_type="slack",
    severity="critical",
    data_asset="users_table",
)

# From checkpoint result
fingerprint = NotificationFingerprint.from_checkpoint_result(
    checkpoint_result,
    action_type="slack",
    include_issues=True,
    include_metadata=False,
)
```

### Window Strategies

| Strategy | Description |
|----------|-------------|
| `SlidingWindowStrategy` | Rolling time window |
| `TumblingWindowStrategy` | Fixed time buckets |
| `SessionWindowStrategy` | Activity-based windows |
| `AdaptiveStrategy` | Adaptive window size |

### Deduplication Stores

```python
from truthound.checkpoint.deduplication.stores import (
    InMemoryDeduplicationStore,
    RedisStreamsDeduplicationStore,
)

# In-memory (development)
store = InMemoryDeduplicationStore(max_size=10000)

# Redis Streams (production)
store = RedisStreamsDeduplicationStore(
    redis_url="redis://localhost:6379",
    stream_prefix="truthound:dedup:",
    max_stream_length=10000,
)
```

## Throttling Configuration

Rate limit notifications.

```python
from truthound.checkpoint.throttling.protocols import (
    ThrottlingConfig,
    RateLimit,
    RateLimitScope,
    TimeUnit,
)

config = ThrottlingConfig(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
    burst_multiplier=1.0,               # Allow burst
    scope=RateLimitScope.GLOBAL,        # Rate limit scope
    algorithm="token_bucket",           # Algorithm
    enabled=True,
    priority_bypass=False,              # Bypass for critical
    priority_threshold="critical",
    queue_on_throttle=False,            # Queue throttled
    max_queue_size=1000,
)

# Custom rate limits
custom_limits = {
    "slack": [
        RateLimit.per_minute(5),
        RateLimit.per_hour(50),
    ],
    "email": [
        RateLimit.per_hour(10),
        RateLimit.per_day(50),
    ],
}
```

**Rate Limit Scopes:**

| Scope | Description |
|-------|-------------|
| `GLOBAL` | All notifications |
| `PER_ACTION` | Per action type |
| `PER_CHECKPOINT` | Per checkpoint |
| `PER_ACTION_CHECKPOINT` | Per action + checkpoint |
| `PER_SEVERITY` | Per severity level |
| `PER_DATA_ASSET` | Per data asset |
| `CUSTOM` | Custom key function |

### Throttlers

```python
from truthound.checkpoint.throttling.throttlers import (
    TokenBucketThrottler,
    FixedWindowThrottler,
    SlidingWindowThrottler,
    CompositeThrottler,
    ThrottlerBuilder,
)

# Token bucket (default)
throttler = TokenBucketThrottler(config)

# Fixed window
throttler = FixedWindowThrottler(config)

# Sliding window
throttler = SlidingWindowThrottler(config)

# Builder pattern
throttler = (
    ThrottlerBuilder()
    .with_limit(RateLimit.per_minute(10))
    .with_limit(RateLimit.per_hour(100))
    .with_scope(RateLimitScope.PER_ACTION)
    .with_burst(1.5)
    .build()
)
```

## Escalation Configuration

Configure multi-level escalation policies.

```python
from truthound.checkpoint.escalation.protocols import (
    EscalationTarget,
    EscalationTrigger,
    TargetType,
)

# Define escalation targets
target = EscalationTarget(
    type=TargetType.USER,
    identifier="@oncall",
    name="On-Call Engineer",
    priority=1,
)

# Factory methods
target = EscalationTarget.user("@john", name="John Doe")
target = EscalationTarget.team("data-team", name="Data Team")
```

**Target Types:**

| Type | Description |
|------|-------------|
| `USER` | Individual user |
| `TEAM` | Team/group |
| `CHANNEL` | Chat channel |
| `SCHEDULE` | On-call schedule |
| `WEBHOOK` | Webhook endpoint |
| `EMAIL` | Email address |
| `PHONE` | Phone number |
| `CUSTOM` | Custom target |

**Escalation Triggers:**

| Trigger | Description |
|---------|-------------|
| `UNACKNOWLEDGED` | Not acknowledged in time |
| `UNRESOLVED` | Not resolved in time |
| `SEVERITY_UPGRADE` | Severity increased |
| `REPEATED_FAILURE` | Multiple failures |
| `THRESHOLD_BREACH` | Threshold exceeded |
| `MANUAL` | Manual escalation |
| `SCHEDULED` | Scheduled escalation |

### Escalation Stores

```python
from truthound.checkpoint.escalation.stores import (
    InMemoryEscalationStore,
    RedisEscalationStore,
    SQLiteEscalationStore,
)

# In-memory
store = InMemoryEscalationStore()

# Redis
store = RedisEscalationStore(
    redis_url="redis://localhost:6379",
    prefix="truthound:escalation:",
)

# SQLite
store = SQLiteEscalationStore(
    db_path="./escalation.db",
)
```

## Complete Example

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.checkpoint.actions import SlackNotification, EmailNotification
from truthound.checkpoint.routing import ActionRouter, Route, RouteMode
from truthound.checkpoint.routing.rules import SeverityRule, AllOf, IssueCountRule

# Configure checkpoint
config = CheckpointConfig(
    name="daily_user_validation",
    data_source="users_table",
    validators=["not_null", "unique", "email_format"],
    fail_on_critical=True,
    timeout_seconds=1800,
    tags={"environment": "production"},
)

# Configure actions
slack = SlackNotification(SlackConfig(
    webhook_url="${SLACK_WEBHOOK}",
    channel="#data-quality",
    notify_on="failure",
))

email = EmailNotification(EmailConfig(
    smtp_host="smtp.example.com",
    to_addresses=["team@example.com"],
    notify_on="failure",
))

# Configure router
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

router.add_route(Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[slack, email],
    priority="critical",
))

router.add_route(Route(
    name="high_volume",
    rule=AllOf([
        SeverityRule(min_severity="high"),
        IssueCountRule(min_issues=10),
    ]),
    actions=[slack],
    priority="high",
))

# Create and run checkpoint
checkpoint = Checkpoint(config)
checkpoint.set_router(router)
result = checkpoint.run()

print(f"Status: {result.status}")
print(f"Issues: {result.validation_result.issue_count}")
```

## Environment Variables

```bash
# Slack
export SLACK_WEBHOOK=https://hooks.slack.com/...

# PagerDuty
export PAGERDUTY_KEY=your-routing-key

# Email
export SMTP_HOST=smtp.example.com
export SMTP_USER=user@example.com
export SMTP_PASSWORD=secret

# Redis (for deduplication/throttling)
export REDIS_URL=redis://localhost:6379
```
