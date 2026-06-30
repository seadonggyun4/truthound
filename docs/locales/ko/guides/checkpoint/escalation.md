# Escalation Policies

실무 운영 가이드에서 Manages, APScheduler-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    EscalationEngine                         │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Policy    │───▶│  Scheduler  │───▶│   Store     │     │
│  │  Manager    │    │ (APScheduler)│    │(InMemory/   │     │
│  │             │    │             │    │ Redis/SQLite)│     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              EscalationStateMachine                  │   │
│  │  PENDING → ACTIVE → ESCALATING → (loop to ACTIVE)   │   │
│  │              ↓           ↓                           │   │
│  │         ACKNOWLEDGED  ACKNOWLEDGED                   │   │
│  │              ↓           ↓                           │   │
│  │           RESOLVED    RESOLVED                       │   │
│  │    Any state → CANCELLED, TIMED_OUT, FAILED         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Types

### TargetType

실무 운영 가이드에서 Escalation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import TargetType

class TargetType(str, Enum):
    USER = "user"         # Individual user
    TEAM = "team"         # Team
    CHANNEL = "channel"   # Channel (Slack, etc.)
    SCHEDULE = "schedule" # On-call schedule
    WEBHOOK = "webhook"   # Webhook URL
    EMAIL = "email"       # Email
    PHONE = "phone"       # Phone
    CUSTOM = "custom"     # Custom
```

### EscalationTrigger

실무 운영 가이드에서 Escalation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import EscalationTrigger

class EscalationTrigger(str, Enum):
    UNACKNOWLEDGED = "unacknowledged"    # Not acknowledged
    UNRESOLVED = "unresolved"            # Not resolved
    SEVERITY_UPGRADE = "severity_upgrade" # Severity upgrade
    REPEATED_FAILURE = "repeated_failure" # Repeated failure
    THRESHOLD_BREACH = "threshold_breach" # Threshold exceeded
    MANUAL = "manual"                     # Manual
    SCHEDULED = "scheduled"               # Scheduled
    CUSTOM = "custom"                     # Custom
```

### EscalationState

실무 운영 가이드에서 Escalation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import EscalationState

class EscalationState(str, Enum):
    PENDING = "pending"         # Initial state, waiting to start
    ACTIVE = "active"           # Currently notifying at level
    ESCALATING = "escalating"   # Escalating to next level
    ACKNOWLEDGED = "acknowledged"  # Responder acknowledged
    RESOLVED = "resolved"       # Issue resolved
    CANCELLED = "cancelled"     # Manually cancelled
    TIMED_OUT = "timed_out"     # Max escalation reached or timed out
    FAILED = "failed"           # System error during escalation
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationTarget

실무 운영 가이드에서 Notification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import EscalationTarget, TargetType

# Constructor
target = EscalationTarget(
    type=TargetType.USER,
    identifier="user-123",
    name="Team Lead",
    priority=1,
    metadata={"slack_id": "U12345"},
)

# Factory methods
user_target = EscalationTarget.user("user-123", "Team Lead")
team_target = EscalationTarget.team("team-abc", "Platform Team")
channel_target = EscalationTarget.channel("#alerts", "Alert Channel")
schedule_target = EscalationTarget.schedule("oncall-123", "Primary On-call")
webhook_target = EscalationTarget.webhook("https://api.example.com/alert")
email_target = EscalationTarget.email("team@company.com", "Team Email")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationLevel

실무 운영 가이드에서 Escalation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import EscalationLevel, EscalationTarget

level1 = EscalationLevel(
    level=1,                      # Level number (1 = first)
    delay_minutes=0,              # Delay time (0 = immediate)
    targets=[                     # Notification targets
        EscalationTarget.user("lead-123", "Team Lead"),
    ],
    repeat_count=2,               # Repeat count (0 = once only)
    repeat_interval_minutes=5,    # Repeat interval (minutes)
    require_ack=True,             # Require acknowledgement
    auto_resolve_minutes=0,       # Auto resolve time (0 = none)
    conditions={},                # Additional conditions
)

level2 = EscalationLevel(
    level=2,
    delay_minutes=15,             # Escalate after 15 minutes
    targets=[
        EscalationTarget.user("manager-456", "Manager"),
        EscalationTarget.team("team-platform", "Platform Team"),
    ],
)

level3 = EscalationLevel(
    level=3,
    delay_minutes=30,             # Escalate after 30 minutes
    targets=[
        EscalationTarget.user("director-789", "Director"),
        EscalationTarget.schedule("oncall-exec", "Executive On-call"),
    ],
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationPolicy

실무 운영 가이드에서 Complete을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationPolicy,
    EscalationLevel,
    EscalationTarget,
    EscalationTrigger,
)

policy = EscalationPolicy(
    name="critical_alerts",
    description="Critical production alerts",
    levels=[
        EscalationLevel(
            level=1,
            delay_minutes=0,
            targets=[EscalationTarget.user("lead", "Team Lead")],
        ),
        EscalationLevel(
            level=2,
            delay_minutes=15,
            targets=[EscalationTarget.user("manager", "Manager")],
        ),
        EscalationLevel(
            level=3,
            delay_minutes=30,
            targets=[EscalationTarget.user("director", "Director")],
        ),
    ],
    enabled=True,
    triggers=[
        EscalationTrigger.UNACKNOWLEDGED,
        EscalationTrigger.REPEATED_FAILURE,
    ],
    severity_filter=["critical", "high"],  # Apply only to critical, high
    max_escalations=5,                      # Maximum escalation count
    cooldown_minutes=60,                    # Cooldown for same incident
    business_hours_only=False,              # Apply 24/7
    timezone="Asia/Seoul",
)

# Properties
policy.max_level                 # Maximum level number
policy.get_level(1)              # Get level 1
policy.get_next_level(1)         # Get next level (level 2)
```

### Business Hours 설정

```python
policy = EscalationPolicy(
    name="business_hours_alerts",
    levels=[...],
    business_hours_only=True,
    business_hours_start=9,      # 9 AM
    business_hours_end=18,       # 6 PM
    business_days=[0, 1, 2, 3, 4],  # Mon-Fri (0=Monday)
    timezone="Asia/Seoul",
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationEngine

실무 운영 가이드에서 Manages을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicy,
)

# Configuration
config = EscalationEngineConfig(
    store_type="memory",           # memory, redis, sqlite
    store_config={},
    check_business_hours=True,
    metrics_enabled=True,
)

# Create engine
engine = EscalationEngine(config)

# Register policy
engine.register_policy(policy)

# Set notification handler
async def notification_handler(record, level, targets):
    """Handler that sends actual notifications."""
    for target in targets:
        if target.type == TargetType.USER:
            await send_slack_dm(target.identifier, record)
        elif target.type == TargetType.CHANNEL:
            await send_slack_channel(target.identifier, record)
    return True

engine.set_notification_handler(notification_handler)

# Start engine
await engine.start()

# Trigger escalation
result = await engine.trigger(
    incident_id="incident-123",
    context={
        "severity": "critical",
        "checkpoint_name": "production_validation",
        "message": "Critical validation failure",
    },
    policy_name="critical_alerts",
)

if result.success:
    print(f"Escalation started: {result.record.id}")
    print(f"Current level: {result.record.current_level}")

# Acknowledge
result = await engine.acknowledge(
    record_id=result.record.id,
    acknowledged_by="user-123",
)

# Resolve
result = await engine.resolve(
    record_id=result.record.id,
    resolved_by="user-123",
)

# Cancel
result = await engine.cancel(
    record_id=result.record.id,
    cancelled_by="user-123",
    reason="False alarm",
)

# Manual escalation
result = await engine.escalate(
    record_id=result.record.id,
    force=True,  # Force regardless of state
)

# Query
record = engine.get_record(record_id)
active_records = engine.get_active_escalations(policy_name="critical_alerts")

# Stop engine
await engine.stop()
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationRecord

실무 운영 가이드에서 Tracks을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
@dataclass
class EscalationRecord:
    id: str                          # Record ID
    incident_id: str                 # External incident ID
    policy_name: str                 # Policy name
    current_level: int = 1           # Current level
    state: str = "pending"           # Current state
    created_at: datetime             # Creation time
    updated_at: datetime             # Update time
    acknowledged_at: datetime | None # Acknowledgement time
    acknowledged_by: str | None      # Acknowledger
    resolved_at: datetime | None     # Resolution time
    resolved_by: str | None          # Resolver
    next_escalation_at: datetime | None  # Next escalation time
    escalation_count: int = 0        # Escalation count
    notification_count: int = 0      # Notification count
    history: list[dict]              # Event history
    context: dict                    # Trigger context
    metadata: dict                   # Metadata

# Properties
record.is_active         # Is active
record.is_acknowledged   # Is acknowledged
record.is_resolved       # Is resolved
record.duration          # Duration (timedelta)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Storage Backends (3 Types)

### InMemoryEscalationStore

실무 운영 가이드에서 In-memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import InMemoryEscalationStore

store = InMemoryEscalationStore()
```

### RedisEscalationStore

실무 운영 가이드에서 Redis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import RedisEscalationStore

store = RedisEscalationStore(
    redis_url="redis://localhost:6379",
    key_prefix="truthound:escalation:",
)
```

### SQLiteEscalationStore

실무 운영 가이드에서 SQLite, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import SQLiteEscalationStore

store = SQLiteEscalationStore(
    db_path="./escalations.db",
)
```

### create_store Factory

```python
from truthound.checkpoint.escalation import create_store

# Create by type
store = create_store("memory")
store = create_store("redis", redis_url="redis://localhost:6379")
store = create_store("sqlite", db_path="./escalations.db")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Scheduler

### Scheduler Types

```python
from truthound.checkpoint.escalation import (
    InMemoryScheduler,
    AsyncioScheduler,
    create_scheduler,
    SchedulerConfig,
)

# InMemory scheduler (for testing)
scheduler = InMemoryScheduler(...)

# Asyncio scheduler
scheduler = AsyncioScheduler(...)

# Create via factory
config = SchedulerConfig(
    scheduler_type="asyncio",
    max_concurrent_jobs=100,
)
scheduler = create_scheduler(config, callback)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Routing 통합

실무 운영 가이드에서 Integrates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationRule,
    EscalationRuleConfig,
    EscalationAction,
    create_escalation_route,
)
from truthound.checkpoint.routing import ActionRouter

# EscalationRule: Conditional escalation
config = EscalationRuleConfig(
    policy_name="critical_alerts",
    severity_filter=["critical", "high"],
    trigger_type="unacknowledged",
)
rule = EscalationRule(config=config)

# EscalationAction: Escalation trigger action
action = EscalationAction(
    engine=engine,
    policy_name="critical_alerts",
)

# Route creation helper
route = create_escalation_route(
    engine=engine,
    policy_name="critical_alerts",
    rule=SeverityRule(min_severity="high"),
)

# Add to router
router.add_route(route)
```

### 통합 with Existing Actions

```python
from truthound.checkpoint.escalation import setup_escalation_with_existing_actions
from truthound.checkpoint.actions import SlackNotification

# Connect existing actions to escalation
slack_action = SlackNotification(webhook_url="...")

setup_escalation_with_existing_actions(
    engine=engine,
    policy_name="critical_alerts",
    level_actions={
        1: [slack_action],
        2: [slack_action, pagerduty_action],
        3: [slack_action, pagerduty_action, email_action],
    },
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## EscalationPolicyManager

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationPolicyManager,
    EscalationPolicyConfig,
)

# Create from config
config = EscalationPolicyConfig(
    default_policy="default",
    global_enabled=True,
    store_type="redis",
    store_config={"redis_url": "redis://localhost:6379"},
    max_concurrent_escalations=1000,
    cleanup_interval_minutes=60,
)

manager = EscalationPolicyManager(config)

# Add policies
manager.add_policy(critical_policy)
manager.add_policy(warning_policy)

# Query policies
policy = manager.get_policy("critical_alerts")
names = manager.list_policies()

# Start
await manager.start()

# Trigger
result = await manager.trigger(
    incident_id="incident-123",
    context={"severity": "critical"},
    policy_name="critical_alerts",
)

# Statistics
stats = manager.get_stats()

# Stop
await manager.stop()
```

### Create from Dictionary

```python
config = {
    "default_policy": "critical_alerts",
    "store_type": "redis",
    "policies": [
        {
            "name": "critical_alerts",
            "levels": [
                {"level": 1, "delay_minutes": 0, "targets": [...]},
                {"level": 2, "delay_minutes": 15, "targets": [...]},
            ],
        }
    ],
}

manager = EscalationPolicyManager.from_dict(config)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Statistics Retrieval

```python
stats = engine.get_stats()

print(f"Total escalations: {stats.total_escalations}")
print(f"Active escalations: {stats.active_escalations}")
print(f"Acknowledged: {stats.acknowledged_count}")
print(f"Resolved: {stats.resolved_count}")
print(f"Timed out: {stats.timed_out_count}")
print(f"Acknowledgment rate: {stats.acknowledgment_rate:.2%}")
print(f"Resolution rate: {stats.resolution_rate:.2%}")
print(f"Avg time to acknowledge: {stats.avg_time_to_acknowledge_seconds}s")
print(f"Avg time to resolve: {stats.avg_time_to_resolve_seconds}s")
print(f"Notifications sent: {stats.notifications_sent}")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Complete Example

```python
import asyncio
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.escalation import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicy,
    EscalationLevel,
    EscalationTarget,
    EscalationTrigger,
)
from truthound.checkpoint.actions import SlackNotification

# Notification handler
async def notification_handler(record, level, targets):
    slack = SlackNotification(webhook_url="${SLACK_WEBHOOK}")
    for target in targets:
        message = f"[Level {level.level}] Escalation for {record.incident_id}"
        # Actual notification logic
        print(f"Notifying {target.name}: {message}")
    return True

# Escalation policy
policy = EscalationPolicy(
    name="production_critical",
    description="Production critical alerts",
    levels=[
        EscalationLevel(
            level=1,
            delay_minutes=0,
            targets=[
                EscalationTarget.user("team-lead", "Team Lead"),
                EscalationTarget.channel("#alerts", "Alerts Channel"),
            ],
            repeat_count=2,
            repeat_interval_minutes=5,
        ),
        EscalationLevel(
            level=2,
            delay_minutes=15,
            targets=[
                EscalationTarget.user("eng-manager", "Engineering Manager"),
                EscalationTarget.schedule("oncall-primary", "Primary On-call"),
            ],
        ),
        EscalationLevel(
            level=3,
            delay_minutes=30,
            targets=[
                EscalationTarget.user("director", "Director of Engineering"),
                EscalationTarget.email("leadership@company.com", "Leadership"),
            ],
        ),
    ],
    triggers=[EscalationTrigger.UNACKNOWLEDGED],
    severity_filter=["critical", "high"],
    cooldown_minutes=60,
)

# Engine configuration and startup
config = EscalationEngineConfig(
    store_type="memory",
    metrics_enabled=True,
)

engine = EscalationEngine(config)
engine.register_policy(policy)
engine.set_notification_handler(notification_handler)


async def main():
    await engine.start()

    # Trigger escalation after checkpoint execution
    checkpoint = Checkpoint(
        name="production_validation",
        data_source="data.csv",
        validators=["null"],
    )
    result = checkpoint.run()

    if result.status.value == "failure":
        escalation_result = await engine.trigger(
            incident_id=f"cp-{result.run_id}",
            context={
                "severity": "critical",
                "checkpoint_name": result.checkpoint_name,
                "issues": result.validation_view.statistics.total_issues if result.validation_view else 0,
            },
            policy_name="production_critical",
        )

        print(f"Escalation triggered: {escalation_result.record.id}")

    await engine.stop()


asyncio.run(main())
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## YAML 설정

```yaml
escalation:
  enabled: true
  default_policy: production_critical
  store:
    type: redis
    redis_url: redis://localhost:6379

  policies:
    - name: production_critical
      description: Production critical alerts
      severity_filter:
        - critical
        - high
      triggers:
        - unacknowledged
      cooldown_minutes: 60
      levels:
        - level: 1
          delay_minutes: 0
          targets:
            - type: user
              identifier: team-lead
              name: Team Lead
            - type: channel
              identifier: "#alerts"
              name: Alerts Channel
          repeat_count: 2
          repeat_interval_minutes: 5

        - level: 2
          delay_minutes: 15
          targets:
            - type: user
              identifier: eng-manager
              name: Engineering Manager
            - type: schedule
              identifier: oncall-primary

        - level: 3
          delay_minutes: 30
          targets:
            - type: user
              identifier: director
            - type: email
              identifier: leadership@company.com
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## State Transition Diagram

```
                    ┌─────────────────────────┐
                    │        PENDING          │
                    └────────────┬────────────┘
                                 │ start()
                                 ▼
        ┌───────────────────────────────────────────────┐
        │                    ACTIVE                     │
        └────┬─────────────┬────────────────┬───────────┘
             │             │                │
             │ ack()       │ timeout        │ escalate()
             ▼             ▼                ▼
    ┌────────────┐  ┌────────────┐  ┌────────────────┐
    │ACKNOWLEDGED│  │ TIMED_OUT  │  │   ESCALATING   │
    └──────┬─────┘  └────────────┘  └───────┬────────┘
           │                                │
           │ resolve()                      │ escalate_success
           ▼                                ▼
    ┌────────────┐                  ┌────────────────┐
    │  RESOLVED  │                  │     ACTIVE     │
    └────────────┘                  │ (next level)   │
                                    └────────────────┘

        cancel() from any state → CANCELLED
        error from any state → FAILED
```
