# Notification Deduplication

알림 중복을 방지하는 시스템입니다. 시간 윈도우 기반으로 동일한 알림이 반복 발송되는 것을 억제합니다.

## 개요

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

## 핵심 클래스

### TimeWindow

중복 제거 시간 윈도우입니다.

```python
from truthound.checkpoint.deduplication import TimeWindow, WindowUnit

class WindowUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

# 5분 윈도우
window = TimeWindow(seconds=300)
window = TimeWindow(minutes=5)

# 1시간 윈도우
window = TimeWindow(hours=1)

# 1일 윈도우
window = TimeWindow(days=1)

# 복합
window = TimeWindow(hours=1, minutes=30)  # 90분
```

### NotificationFingerprint

알림의 고유 식별자입니다.

```python
@dataclass
class NotificationFingerprint:
    """알림 핑거프린트."""
    key: str                  # 해시 키 (고유 식별자)
    checkpoint_name: str      # 체크포인트 이름
    action_type: str          # 액션 타입 (slack, email 등)
    components: dict[str, Any] = field(default_factory=dict)  # 핑거프린트 생성에 사용된 컴포넌트
    created_at: datetime = field(default_factory=datetime.now)  # 생성 시간
    metadata: dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터

    @classmethod
    def generate(
        cls,
        checkpoint_name: str,
        action_type: str,
        *,
        severity: str | None = None,      # 심각도 (선택)
        data_asset: str | None = None,    # 데이터 자산 (선택)
        issue_types: Sequence[str] | None = None,  # 이슈 타입들 (선택)
        custom_key: str | None = None,    # 커스텀 키 (선택)
        algorithm: str = "sha256",        # 해시 알고리즘
        **extra_components: Any,
    ) -> "NotificationFingerprint": ...
```

### DeduplicationPolicy

중복 제거 정책입니다.

```python
from truthound.checkpoint.deduplication import DeduplicationPolicy

class DeduplicationPolicy(str, Enum):
    NONE = "none"           # 중복 제거 안 함
    BASIC = "basic"         # checkpoint + action_type으로 구분
    SEVERITY = "severity"   # + severity로 구분
    ISSUE_BASED = "issue_based"  # + issue types로 구분
    STRICT = "strict"       # 전체 핑거프린트 사용
    CUSTOM = "custom"       # 사용자 정의 함수
```

---

## 기본 사용법

```python
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    DeduplicationConfig,
    InMemoryDeduplicationStore,
    TimeWindow,
    DeduplicationPolicy,
)

# Deduplication 설정
config = DeduplicationConfig(
    policy=DeduplicationPolicy.SEVERITY,
    default_window=TimeWindow(minutes=5),
)

# Deduplicator 생성
deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    config=config,
)

# 중복 확인 (checkpoint_result와 action_type으로 확인)
result = deduplicator.check(checkpoint_result, "slack", severity="high")

if result.should_send:
    # 알림 발송
    await action.execute(checkpoint_result)
    # 발송 기록
    deduplicator.mark_sent(result.fingerprint)
else:
    print(f"Notification suppressed: {result.message}")

# 또는 간단히 is_duplicate 사용
if not deduplicator.is_duplicate(checkpoint_result, "slack", severity="high"):
    await action.execute(checkpoint_result)
```

---

## 스토어 백엔드

### InMemoryDeduplicationStore

단일 프로세스용 인메모리 스토어입니다.

```python
from truthound.checkpoint.deduplication import InMemoryDeduplicationStore

store = InMemoryDeduplicationStore(
    max_size=10000,  # 최대 레코드 수
    auto_cleanup_interval=60,  # 자동 정리 주기 (초)
)

deduplicator = NotificationDeduplicator(store=store)
```

### RedisStreamsDeduplicationStore

분산 환경용 Redis 스토어입니다.

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

## 윈도우 전략 (4가지)

### SlidingWindowStrategy

고정 시간 윈도우입니다. 윈도우 내 모든 알림을 억제합니다.

```python
from truthound.checkpoint.deduplication import SlidingWindowStrategy, TimeWindow

strategy = SlidingWindowStrategy(
    window=TimeWindow(minutes=5),
)

# 예:
# 10:00:00 - 첫 알림 → 허용
# 10:02:00 - 동일 알림 → 억제 (5분 이내)
# 10:06:00 - 동일 알림 → 허용 (5분 지남)
```

### TumblingWindowStrategy

겹치지 않는 고정 버킷입니다.

```python
from truthound.checkpoint.deduplication import TumblingWindowStrategy

strategy = TumblingWindowStrategy(
    bucket_size=TimeWindow(minutes=15),
)

# 예: 15분 버킷
# 10:00-10:15 버킷: 첫 알림만 허용
# 10:15-10:30 버킷: 새 버킷, 첫 알림 허용
```

### SessionWindowStrategy

이벤트 기반 세션입니다. 일정 시간 알림이 없으면 세션 종료.

```python
from truthound.checkpoint.deduplication import SessionWindowStrategy

strategy = SessionWindowStrategy(
    gap=TimeWindow(minutes=10),  # 10분간 알림 없으면 새 세션
)

# 예:
# 10:00 - 알림 → 세션 시작, 허용
# 10:05 - 알림 → 세션 내, 억제
# 10:20 - 알림 → 10분 초과, 새 세션, 허용
```

### AdaptiveWindowStrategy (미구현 예정)

알림 빈도에 따라 동적으로 윈도우 크기를 조절합니다.

---

## 액션별 윈도우 설정

액션 타입별로 다른 윈도우를 설정할 수 있습니다.

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
    # 액션별 윈도우
    action_windows={
        "pagerduty": TimeWindow(hours=1),     # PagerDuty: 1시간
        "slack": TimeWindow(minutes=5),        # Slack: 5분
        "email": TimeWindow(hours=24),         # Email: 24시간
    },
    # Severity별 윈도우
    severity_windows={
        "critical": TimeWindow(minutes=1),     # Critical: 1분
        "high": TimeWindow(minutes=5),         # High: 5분
        "medium": TimeWindow(minutes=15),      # Medium: 15분
    },
)

deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    config=config,
)
```

---

## Middleware 사용

액션을 자동으로 감싸서 중복 제거를 적용합니다.

```python
from truthound.checkpoint.deduplication import (
    DeduplicationMiddleware,
    deduplicated,
)
from truthound.checkpoint.actions import SlackNotification

# Middleware 사용
middleware = DeduplicationMiddleware(
    deduplicator=deduplicator,
)

slack_action = SlackNotification(webhook_url="...")
deduplicated_action = middleware.wrap(slack_action)

# 또는 데코레이터 사용
@deduplicated(window=TimeWindow(minutes=5))
async def send_notification(result):
    # 알림 로직
    pass
```

---

## 통계 조회

```python
# 중복 제거 통계
stats = deduplicator.get_stats()

print(f"Total evaluated: {stats.total_evaluated}")
print(f"Total suppressed: {stats.suppressed}")
print(f"Suppression ratio: {stats.suppression_ratio:.2%}")
print(f"Active fingerprints: {stats.active_fingerprints}")
```

---

## 전체 예시

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

# Deduplication 설정
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

# Middleware로 액션 래핑
middleware = DeduplicationMiddleware(deduplicator=deduplicator)

slack_action = middleware.wrap(
    SlackNotification(webhook_url="${SLACK_WEBHOOK}")
)

pagerduty_action = middleware.wrap(
    PagerDutyAction(routing_key="${PAGERDUTY_KEY}")
)

# Checkpoint에 적용
checkpoint = Checkpoint(
    name="production_check",
    data_source="data.csv",
    validators=["null"],
    actions=[slack_action, pagerduty_action],
)
```

---

## YAML 설정

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
    type: redis  # 또는 memory
    redis_url: redis://localhost:6379
```
