# Throttling (Rate Limiting)

알림 발송 빈도를 제어하는 Rate Limiting 시스템입니다. Token Bucket 알고리즘을 기반으로 분/시간/일 단위의 다중 레벨 제한을 지원합니다.

## 개요

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
           retry_after: N초
```

## 핵심 타입

### TimeUnit

시간 단위입니다.

```python
from truthound.checkpoint.throttling import TimeUnit

class TimeUnit(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
```

### RateLimitScope

Rate limit 적용 범위입니다.

```python
from truthound.checkpoint.throttling import RateLimitScope

class RateLimitScope(str, Enum):
    GLOBAL = "global"                    # 모든 알림 통합
    PER_ACTION = "per_action"            # 액션 타입별 (slack, email 등)
    PER_CHECKPOINT = "per_checkpoint"    # 체크포인트별
    PER_ACTION_CHECKPOINT = "per_action_checkpoint"  # 액션 + 체크포인트
    PER_SEVERITY = "per_severity"        # Severity별
    PER_DATA_ASSET = "per_data_asset"    # 데이터 자산별
    CUSTOM = "custom"                    # 커스텀 키
```

### ThrottleStatus

Throttle 결과 상태입니다.

```python
from truthound.checkpoint.throttling import ThrottleStatus

class ThrottleStatus(str, Enum):
    ALLOWED = "allowed"          # 요청 허용
    THROTTLED = "throttled"      # Rate limit 초과로 거부
    QUEUED = "queued"            # 큐에 대기 (추후 처리)
    BURST_ALLOWED = "burst_allowed"  # Burst 용량으로 허용
    ERROR = "error"              # 오류 발생
```

### RateLimit

Rate limit 설정입니다.

```python
from truthound.checkpoint.throttling import RateLimit, TimeUnit

@dataclass(frozen=True)
class RateLimit:
    limit: int                  # 최대 요청 수
    time_unit: TimeUnit         # 시간 단위
    burst_multiplier: float = 1.0  # Burst 용량 배율 (1.0 = 버스트 없음)

# 팩토리 메서드
limit = RateLimit.per_minute(10, burst_multiplier=1.5)  # 분당 10회, 버스트 15회
limit = RateLimit.per_hour(100)                          # 시간당 100회
limit = RateLimit.per_day(500)                           # 일일 500회

# 속성
limit.window_seconds      # 윈도우 크기 (초)
limit.burst_limit         # 버스트 한도 (limit * multiplier)
limit.tokens_per_second   # 토큰 초당 보충률
```

### ThrottleResult

Throttle 확인 결과입니다.

```python
@dataclass
class ThrottleResult:
    status: ThrottleStatus       # 결과 상태
    key: ThrottlingKey           # Throttling 키
    allowed: bool                # 허용 여부
    retry_after: float = 0.0     # 재시도 대기 시간 (초)
    remaining: int = 0           # 남은 토큰/요청 수
    limit: RateLimit | None = None  # 적용된 Rate limit
    message: str = ""            # 메시지
    metadata: dict = field(default_factory=dict)  # 추가 메타데이터
```

---

## Throttler 타입 (5가지)

### TokenBucketThrottler

Token Bucket 알고리즘입니다. 버스트 허용 후 일정 속도로 토큰을 보충합니다.

```python
from truthound.checkpoint.throttling import (
    TokenBucketThrottler,
    RateLimit,
    ThrottlingKey,
    TimeUnit,
)

throttler = TokenBucketThrottler("api")

# Rate limit 정의
limit = RateLimit.per_minute(10, burst_multiplier=1.5)  # 분당 10, 버스트 15

# Throttling 키 생성
key = ThrottlingKey.for_global(TimeUnit.MINUTE)

# 확인만 (토큰 소비 안 함)
result = throttler.check(key, limit)
if result.allowed:
    print(f"Remaining: {result.remaining}")

# 토큰 획득 (확인 + 소비)
result = throttler.acquire(key, limit)
if result.allowed:
    send_notification()
else:
    print(f"Retry after {result.retry_after:.1f}s")
```

**특징:**
- 순간적인 버스트 트래픽 허용
- 일정 속도로 토큰 보충
- 스무스한 rate limiting

### SlidingWindowThrottler

슬라이딩 윈도우 알고리즘입니다. 더 정확한 rate limiting을 제공합니다.

```python
from truthound.checkpoint.throttling import SlidingWindowThrottler

throttler = SlidingWindowThrottler("api")
limit = RateLimit.per_hour(100)
key = ThrottlingKey.for_action("slack", TimeUnit.HOUR)

result = throttler.acquire(key, limit)
```

**특징:**
- 윈도우 경계에서 급격한 변동 없음
- 더 일관된 rate limiting
- 메모리 사용량 약간 높음

### FixedWindowThrottler

고정 윈도우 알고리즘입니다. 단순하지만 윈도우 경계에서 2배 트래픽 가능.

```python
from truthound.checkpoint.throttling import FixedWindowThrottler

throttler = FixedWindowThrottler("api")
limit = RateLimit.per_minute(10)
key = ThrottlingKey.for_global(TimeUnit.MINUTE)

result = throttler.acquire(key, limit)
```

**특징:**
- 구현이 간단
- 메모리 효율적
- 윈도우 경계에서 최대 2x 트래픽 가능 (edge effect)

### CompositeThrottler

여러 Rate limit을 조합합니다. 모든 limit을 통과해야 허용.

```python
from truthound.checkpoint.throttling import CompositeThrottler, RateLimit

throttler = CompositeThrottler("multi-level", algorithm="token_bucket")

# 다중 레벨 Rate limit 추가
throttler.add_limit(RateLimit.per_minute(10))
throttler.add_limit(RateLimit.per_hour(100))
throttler.add_limit(RateLimit.per_day(500))

# 또는 한 번에 설정
throttler.with_limits([
    RateLimit.per_minute(10),
    RateLimit.per_hour(100),
    RateLimit.per_day(500),
])

key = ThrottlingKey.for_global(TimeUnit.MINUTE)
result = throttler.acquire(key)  # 모든 limit 확인
```

**특징:**
- 분/시간/일 다중 제한
- 모든 limit 통과 필요
- 가장 엄격한 limit이 최종 결정

### NoOpThrottler

항상 허용하는 패스스루 Throttler입니다. 테스트 또는 비활성화 용도.

```python
from truthound.checkpoint.throttling import NoOpThrottler

throttler = NoOpThrottler()
result = throttler.acquire(key, limit)
assert result.allowed  # 항상 True
```

---

## NotificationThrottler

고수준 Throttling 서비스입니다.

```python
from truthound.checkpoint.throttling import (
    NotificationThrottler,
    ThrottlingConfig,
    RateLimitScope,
)

# Config로 생성
config = ThrottlingConfig(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
    burst_multiplier=1.5,
    scope=RateLimitScope.PER_ACTION,
    algorithm="token_bucket",
    enabled=True,
    priority_bypass=True,         # 중요 알림 bypass
    priority_threshold="critical",  # critical은 bypass
)

throttler = NotificationThrottler(config=config)

# 확인
result = throttler.check(
    action_type="slack",
    checkpoint_name="data_quality",
    severity="high",
)

# 획득
result = throttler.acquire(
    action_type="slack",
    checkpoint_name="data_quality",
    severity="high",
)

if result.allowed:
    send_notification()

# CheckpointResult로 확인
result = throttler.acquire_result(
    checkpoint_result=checkpoint_result,
    action_type="slack",
)

# 편의 메서드
if throttler.is_throttled("slack", "my_checkpoint"):
    print("Rate limit exceeded")
```

### ThrottlingConfig

```python
@dataclass
class ThrottlingConfig:
    per_minute_limit: int | None = 10    # 분당 한도
    per_hour_limit: int | None = 100     # 시간당 한도
    per_day_limit: int | None = 500      # 일일 한도
    burst_multiplier: float = 1.0        # 버스트 배율
    scope: RateLimitScope = RateLimitScope.GLOBAL  # 적용 범위
    algorithm: str = "token_bucket"      # 알고리즘
    enabled: bool = True                 # 활성화
    custom_limits: dict[str, list[RateLimit]] = {}  # 액션별 커스텀 한도
    severity_limits: dict[str, list[RateLimit]] = {}  # Severity별 한도
    priority_bypass: bool = False        # 우선순위 bypass
    priority_threshold: str = "critical" # bypass 임계값
    queue_on_throttle: bool = False      # 큐 대기 여부
    max_queue_size: int = 1000           # 최대 큐 크기
```

---

## ThrottlerBuilder

Fluent API로 Throttler를 구성합니다.

```python
from truthound.checkpoint.throttling import ThrottlerBuilder, RateLimitScope

throttler = (
    ThrottlerBuilder()
    # 기본 한도
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .with_per_day_limit(500)
    # 버스트 설정
    .with_burst_allowance(1.5)
    # 알고리즘 및 범위
    .with_algorithm("token_bucket")  # sliding_window, fixed_window
    .with_scope(RateLimitScope.PER_ACTION)
    # 우선순위 bypass (critical 알림은 무시)
    .with_priority_bypass("critical")
    # 액션별 커스텀 한도
    .with_action_limit("pagerduty", per_minute=5, per_hour=20)
    .with_action_limit("slack", per_minute=20, per_hour=200)
    # Severity별 한도
    .with_severity_limit("info", per_minute=5, per_hour=50)
    .with_severity_limit("critical", per_minute=None, per_hour=None)  # 제한 없음
    # 큐잉 활성화
    .with_queueing(max_size=1000)
    .build()
)
```

---

## Middleware

액션을 자동으로 감싸서 throttling을 적용합니다.

### ThrottlingMiddleware

```python
from truthound.checkpoint.throttling import (
    ThrottlingMiddleware,
    ThrottlerBuilder,
)
from truthound.checkpoint.actions import SlackNotification

# Throttler 생성
throttler = (
    ThrottlerBuilder()
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .build()
)

# Middleware 생성
middleware = ThrottlingMiddleware(throttler=throttler)

# 액션 래핑
slack_action = SlackNotification(webhook_url="...")
throttled_action = middleware.wrap(slack_action)

# 사용 - throttling 자동 적용
await throttled_action.execute(checkpoint_result)
```

### @throttled 데코레이터

```python
from truthound.checkpoint.throttling import throttled, RateLimit

@throttled(
    limits=[
        RateLimit.per_minute(10),
        RateLimit.per_hour(100),
    ]
)
async def send_notification(result):
    # 알림 로직
    pass
```

### 전역 Throttling

```python
from truthound.checkpoint.throttling import (
    configure_global_throttling,
    get_global_middleware,
)

# 전역 설정
configure_global_throttling(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
)

# 전역 middleware 사용
middleware = get_global_middleware()
throttled_action = middleware.wrap(slack_action)
```

---

## Storage Backend

### InMemoryThrottlingStore

단일 프로세스용 인메모리 스토어입니다.

```python
from truthound.checkpoint.throttling import InMemoryThrottlingStore

store = InMemoryThrottlingStore()

# NotificationThrottler에서 내부적으로 사용
```

---

## 통계 조회

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

## 전체 예시

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.throttling import (
    ThrottlerBuilder,
    ThrottlingMiddleware,
    RateLimitScope,
)
from truthound.checkpoint.actions import SlackNotification, PagerDutyAction

# 다중 레벨 Throttler
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

# 액션 래핑
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
      per_minute: null  # 제한 없음
```

---

## Throttler 비교

| Throttler | 용도 | 버스트 | 정확성 | 메모리 |
|-----------|------|--------|--------|--------|
| **TokenBucket** | 일반 | 허용 | 중간 | 낮음 |
| **SlidingWindow** | 정확한 제한 | 없음 | 높음 | 중간 |
| **FixedWindow** | 단순 제한 | 2x 가능 | 낮음 | 낮음 |
| **Composite** | 다중 제한 | 설정별 | 높음 | 중간 |
| **NoOp** | 테스트 | N/A | N/A | 없음 |
