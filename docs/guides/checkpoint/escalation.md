# Escalation Policies

다단계 알림 에스컬레이션 정책 시스템입니다. APScheduler 기반 스케줄링과 상태 머신으로 알림의 수명 주기를 관리합니다.

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

## 핵심 타입

### TargetType

에스컬레이션 대상 타입입니다.

```python
from truthound.checkpoint.escalation import TargetType

class TargetType(str, Enum):
    USER = "user"         # 개별 사용자
    TEAM = "team"         # 팀
    CHANNEL = "channel"   # 채널 (Slack 등)
    SCHEDULE = "schedule" # On-call 스케줄
    WEBHOOK = "webhook"   # Webhook URL
    EMAIL = "email"       # 이메일
    PHONE = "phone"       # 전화
    CUSTOM = "custom"     # 커스텀
```

### EscalationTrigger

에스컬레이션 트리거 조건입니다.

```python
from truthound.checkpoint.escalation import EscalationTrigger

class EscalationTrigger(str, Enum):
    UNACKNOWLEDGED = "unacknowledged"    # 미확인
    UNRESOLVED = "unresolved"            # 미해결
    SEVERITY_UPGRADE = "severity_upgrade" # Severity 상향
    REPEATED_FAILURE = "repeated_failure" # 반복 실패
    THRESHOLD_BREACH = "threshold_breach" # 임계값 초과
    MANUAL = "manual"                     # 수동
    SCHEDULED = "scheduled"               # 스케줄
    CUSTOM = "custom"                     # 커스텀
```

### EscalationState

에스컬레이션 상태입니다.

```python
from truthound.checkpoint.escalation import EscalationState

class EscalationState(str, Enum):
    PENDING = "pending"         # 초기 상태, 시작 대기
    ACTIVE = "active"           # 현재 레벨 알림 중
    ESCALATING = "escalating"   # 다음 레벨로 에스컬레이션 중
    ACKNOWLEDGED = "acknowledged"  # 응답자가 확인함
    RESOLVED = "resolved"       # 문제 해결됨
    CANCELLED = "cancelled"     # 수동 취소됨
    TIMED_OUT = "timed_out"     # 최대 에스컬레이션 도달 또는 타임아웃
    FAILED = "failed"           # 에스컬레이션 중 시스템 에러
```

---

## EscalationTarget

알림 대상 정의입니다.

```python
from truthound.checkpoint.escalation import EscalationTarget, TargetType

# 생성자
target = EscalationTarget(
    type=TargetType.USER,
    identifier="user-123",
    name="Team Lead",
    priority=1,
    metadata={"slack_id": "U12345"},
)

# 팩토리 메서드
user_target = EscalationTarget.user("user-123", "Team Lead")
team_target = EscalationTarget.team("team-abc", "Platform Team")
channel_target = EscalationTarget.channel("#alerts", "Alert Channel")
schedule_target = EscalationTarget.schedule("oncall-123", "Primary On-call")
webhook_target = EscalationTarget.webhook("https://api.example.com/alert")
email_target = EscalationTarget.email("team@company.com", "Team Email")
```

---

## EscalationLevel

에스컬레이션 레벨 정의입니다.

```python
from truthound.checkpoint.escalation import EscalationLevel, EscalationTarget

level1 = EscalationLevel(
    level=1,                      # 레벨 번호 (1 = 첫 번째)
    delay_minutes=0,              # 지연 시간 (0 = 즉시)
    targets=[                     # 알림 대상
        EscalationTarget.user("lead-123", "Team Lead"),
    ],
    repeat_count=2,               # 반복 횟수 (0 = 1회만)
    repeat_interval_minutes=5,    # 반복 간격 (분)
    require_ack=True,             # 확인 필수 여부
    auto_resolve_minutes=0,       # 자동 해결 시간 (0 = 없음)
    conditions={},                # 추가 조건
)

level2 = EscalationLevel(
    level=2,
    delay_minutes=15,             # 15분 후 에스컬레이션
    targets=[
        EscalationTarget.user("manager-456", "Manager"),
        EscalationTarget.team("team-platform", "Platform Team"),
    ],
)

level3 = EscalationLevel(
    level=3,
    delay_minutes=30,             # 30분 후 에스컬레이션
    targets=[
        EscalationTarget.user("director-789", "Director"),
        EscalationTarget.schedule("oncall-exec", "Executive On-call"),
    ],
)
```

---

## EscalationPolicy

에스컬레이션 정책 전체 정의입니다.

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
    severity_filter=["critical", "high"],  # critical, high만 적용
    max_escalations=5,                      # 최대 에스컬레이션 횟수
    cooldown_minutes=60,                    # 동일 인시던트 쿨다운
    business_hours_only=False,              # 24시간 적용
    timezone="Asia/Seoul",
)

# 속성
policy.max_level                 # 최대 레벨 번호
policy.get_level(1)              # 레벨 1 가져오기
policy.get_next_level(1)         # 다음 레벨 (레벨 2) 가져오기
```

### Business Hours 설정

```python
policy = EscalationPolicy(
    name="business_hours_alerts",
    levels=[...],
    business_hours_only=True,
    business_hours_start=9,      # 오전 9시
    business_hours_end=18,       # 오후 6시
    business_days=[0, 1, 2, 3, 4],  # 월-금 (0=월요일)
    timezone="Asia/Seoul",
)
```

---

## EscalationEngine

에스컬레이션 엔진입니다. 전체 수명 주기를 관리합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicy,
)

# 설정
config = EscalationEngineConfig(
    store_type="memory",           # memory, redis, sqlite
    store_config={},
    check_business_hours=True,
    metrics_enabled=True,
)

# 엔진 생성
engine = EscalationEngine(config)

# 정책 등록
engine.register_policy(policy)

# 알림 핸들러 설정
async def notification_handler(record, level, targets):
    """실제 알림을 발송하는 핸들러."""
    for target in targets:
        if target.type == TargetType.USER:
            await send_slack_dm(target.identifier, record)
        elif target.type == TargetType.CHANNEL:
            await send_slack_channel(target.identifier, record)
    return True

engine.set_notification_handler(notification_handler)

# 엔진 시작
await engine.start()

# 에스컬레이션 트리거
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

# 확인 (Acknowledge)
result = await engine.acknowledge(
    record_id=result.record.id,
    acknowledged_by="user-123",
)

# 해결 (Resolve)
result = await engine.resolve(
    record_id=result.record.id,
    resolved_by="user-123",
)

# 취소 (Cancel)
result = await engine.cancel(
    record_id=result.record.id,
    cancelled_by="user-123",
    reason="False alarm",
)

# 수동 에스컬레이션
result = await engine.escalate(
    record_id=result.record.id,
    force=True,  # 상태 무시하고 강제
)

# 조회
record = engine.get_record(record_id)
active_records = engine.get_active_escalations(policy_name="critical_alerts")

# 엔진 중지
await engine.stop()
```

---

## EscalationRecord

에스컬레이션 레코드입니다. 진행 상태를 추적합니다.

```python
@dataclass
class EscalationRecord:
    id: str                          # 레코드 ID
    incident_id: str                 # 외부 인시던트 ID
    policy_name: str                 # 정책 이름
    current_level: int = 1           # 현재 레벨
    state: str = "pending"           # 현재 상태
    created_at: datetime             # 생성 시간
    updated_at: datetime             # 갱신 시간
    acknowledged_at: datetime | None # 확인 시간
    acknowledged_by: str | None      # 확인자
    resolved_at: datetime | None     # 해결 시간
    resolved_by: str | None          # 해결자
    next_escalation_at: datetime | None  # 다음 에스컬레이션 시간
    escalation_count: int = 0        # 에스컬레이션 횟수
    notification_count: int = 0      # 알림 발송 횟수
    history: list[dict]              # 이벤트 히스토리
    context: dict                    # 트리거 컨텍스트
    metadata: dict                   # 메타데이터

# 속성
record.is_active         # 활성 여부
record.is_acknowledged   # 확인 여부
record.is_resolved       # 해결 여부
record.duration          # 지속 시간 (timedelta)
```

---

## Storage Backend (3가지)

### InMemoryEscalationStore

단일 프로세스용 인메모리 스토어입니다.

```python
from truthound.checkpoint.escalation import InMemoryEscalationStore

store = InMemoryEscalationStore()
```

### RedisEscalationStore

분산 환경용 Redis 스토어입니다.

```python
from truthound.checkpoint.escalation import RedisEscalationStore

store = RedisEscalationStore(
    redis_url="redis://localhost:6379",
    key_prefix="truthound:escalation:",
)
```

### SQLiteEscalationStore

영구 저장용 SQLite 스토어입니다.

```python
from truthound.checkpoint.escalation import SQLiteEscalationStore

store = SQLiteEscalationStore(
    db_path="./escalations.db",
)
```

### create_store 팩토리

```python
from truthound.checkpoint.escalation import create_store

# 타입별 생성
store = create_store("memory")
store = create_store("redis", redis_url="redis://localhost:6379")
store = create_store("sqlite", db_path="./escalations.db")
```

---

## Scheduler

### 스케줄러 타입

```python
from truthound.checkpoint.escalation import (
    InMemoryScheduler,
    AsyncioScheduler,
    create_scheduler,
    SchedulerConfig,
)

# InMemory 스케줄러 (테스트용)
scheduler = InMemoryScheduler(...)

# Asyncio 스케줄러
scheduler = AsyncioScheduler(...)

# 팩토리로 생성
config = SchedulerConfig(
    scheduler_type="asyncio",
    max_concurrent_jobs=100,
)
scheduler = create_scheduler(config, callback)
```

---

## Routing 통합

라우팅 시스템과 연동합니다.

```python
from truthound.checkpoint.escalation import (
    EscalationRule,
    EscalationRuleConfig,
    EscalationAction,
    create_escalation_route,
)
from truthound.checkpoint.routing import ActionRouter

# EscalationRule: 조건부 에스컬레이션
config = EscalationRuleConfig(
    policy_name="critical_alerts",
    severity_filter=["critical", "high"],
    trigger_type="unacknowledged",
)
rule = EscalationRule(config=config)

# EscalationAction: 에스컬레이션 트리거 액션
action = EscalationAction(
    engine=engine,
    policy_name="critical_alerts",
)

# Route 생성 헬퍼
route = create_escalation_route(
    engine=engine,
    policy_name="critical_alerts",
    rule=SeverityRule(min_severity="high"),
)

# 라우터에 추가
router.add_route(route)
```

### 기존 액션과 통합

```python
from truthound.checkpoint.escalation import setup_escalation_with_existing_actions
from truthound.checkpoint.actions import SlackNotification

# 기존 액션을 에스컬레이션에 연결
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

---

## EscalationPolicyManager

여러 정책을 관리하는 고수준 매니저입니다.

```python
from truthound.checkpoint.escalation import (
    EscalationPolicyManager,
    EscalationPolicyConfig,
)

# Config로 생성
config = EscalationPolicyConfig(
    default_policy="default",
    global_enabled=True,
    store_type="redis",
    store_config={"redis_url": "redis://localhost:6379"},
    max_concurrent_escalations=1000,
    cleanup_interval_minutes=60,
)

manager = EscalationPolicyManager(config)

# 정책 추가
manager.add_policy(critical_policy)
manager.add_policy(warning_policy)

# 정책 조회
policy = manager.get_policy("critical_alerts")
names = manager.list_policies()

# 시작
await manager.start()

# 트리거
result = await manager.trigger(
    incident_id="incident-123",
    context={"severity": "critical"},
    policy_name="critical_alerts",
)

# 통계
stats = manager.get_stats()

# 중지
await manager.stop()
```

### 딕셔너리에서 생성

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

---

## 통계 조회

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

---

## 전체 예시

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

# 알림 핸들러
async def notification_handler(record, level, targets):
    slack = SlackNotification(webhook_url="${SLACK_WEBHOOK}")
    for target in targets:
        message = f"[Level {level.level}] Escalation for {record.incident_id}"
        # 실제 알림 발송 로직
        print(f"Notifying {target.name}: {message}")
    return True

# 에스컬레이션 정책
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

# 엔진 설정 및 시작
config = EscalationEngineConfig(
    store_type="memory",
    metrics_enabled=True,
)

engine = EscalationEngine(config)
engine.register_policy(policy)
engine.set_notification_handler(notification_handler)


async def main():
    await engine.start()

    # Checkpoint 실행 후 에스컬레이션 트리거
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
                "issues": result.validation_result.statistics.total_issues,
            },
            policy_name="production_critical",
        )

        print(f"Escalation triggered: {escalation_result.record.id}")

    await engine.stop()


asyncio.run(main())
```

---

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

---

## 상태 전이 다이어그램

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

        어디서든 cancel() 호출 시 → CANCELLED
        어디서든 error 발생 시 → FAILED
```
