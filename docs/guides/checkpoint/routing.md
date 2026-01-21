# Rule-based Routing

Rule-based Routing은 검증 결과의 조건에 따라 다른 액션을 실행하도록 하는 시스템입니다. Python 표현식과 Jinja2 템플릿 엔진을 지원합니다.

## 개요

```
CheckpointResult
       │
       ▼
┌──────────────────┐
│   ActionRouter   │
│  (RouteMode)     │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Route 1│ │Route 2│ ...
│(Rule) │ │(Rule) │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
[Actions]  [Actions]
```

## 핵심 클래스

### RouteContext

라우팅 규칙 평가에 사용되는 컨텍스트 데이터입니다.

```python
@dataclass(frozen=True)
class RouteContext:
    """라우팅 컨텍스트 (불변)."""
    checkpoint_name: str           # 체크포인트 이름
    run_id: str                    # 실행 ID
    status: str                    # 결과 상태
    data_asset: str                # 데이터 자산
    run_time: datetime             # 실행 시간
    total_issues: int = 0          # 총 이슈 수
    critical_issues: int = 0       # Critical 이슈 수
    high_issues: int = 0           # High 이슈 수
    medium_issues: int = 0         # Medium 이슈 수
    low_issues: int = 0            # Low 이슈 수
    info_issues: int = 0           # Info 이슈 수
    pass_rate: float = 100.0       # 통과율 (0-100)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_duration_ms: float = 0.0  # 검증 소요 시간 (ms)
    error: str | None = None       # 에러 메시지
```

### ActionRouter

라우팅을 관리하는 메인 클래스입니다.

```python
from truthound.checkpoint.routing import ActionRouter, Route
from truthound.checkpoint.routing.base import RouteMode

class RouteMode(str, Enum):
    """라우팅 모드."""
    FIRST_MATCH = "first_match"      # 첫 번째 매칭 라우트만 실행
    ALL_MATCHES = "all_matches"      # 모든 매칭 라우트 실행
    PRIORITY_GROUP = "priority_group" # 가장 높은 우선순위 그룹 실행

# Router 생성
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# Route 추가
router.add_route(route)
```

### Route

규칙과 액션의 매핑입니다.

```python
from truthound.checkpoint.routing import Route
from truthound.checkpoint.routing.base import RoutePriority

class RoutePriority(int, Enum):
    """라우트 우선순위."""
    CRITICAL = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20
    DEFAULT = 0

route = Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=RoutePriority.CRITICAL,  # 또는 정수값
)
```

---

## 내장 규칙 (11개)

### AlwaysRule / NeverRule

항상 매칭하거나 절대 매칭하지 않는 규칙입니다.

```python
from truthound.checkpoint.routing.rules import AlwaysRule, NeverRule

# 기본 라우트용
always = AlwaysRule()  # 항상 True

# 라우트 비활성화용
never = NeverRule()  # 항상 False
```

### SeverityRule

이슈의 심각도에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import SeverityRule

# Critical 이슈가 있으면 매칭
rule = SeverityRule(min_severity="critical")

# High 이상 이슈가 5개 이상이면 매칭
rule = SeverityRule(min_severity="high", min_count=5)

# Medium 이슈만 (범위 지정)
rule = SeverityRule(min_severity="medium", max_severity="medium")

# 정확히 3개의 Critical 이슈
rule = SeverityRule(min_severity="critical", exact_count=3)
```

**Severity 순서**: `critical` > `high` > `medium` > `low` > `info`

### IssueCountRule

이슈 개수에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import IssueCountRule

# 10개 이상의 이슈
rule = IssueCountRule(min_issues=10)

# 5-20개 사이의 이슈
rule = IssueCountRule(min_issues=5, max_issues=20)

# Critical 이슈 3개 이상
rule = IssueCountRule(min_issues=3, count_type="critical")
```

**count_type**: `total`, `critical`, `high`, `medium`, `low`, `info`

### StatusRule

체크포인트 상태에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import StatusRule

# 실패 또는 에러
rule = StatusRule(statuses=["failure", "error"])

# 성공이 아닌 모든 상태
rule = StatusRule(statuses=["success"], negate=True)
```

### TagRule

태그의 존재 또는 값에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import TagRule

# env=prod 태그
rule = TagRule(tags={"env": "prod"})

# env=prod AND team=data
rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=True)

# env=prod OR team=data
rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=False)

# 'critical' 태그 존재 여부 (값 무관)
rule = TagRule(tags={"critical": None})

# 태그가 없으면 매칭
rule = TagRule(tags={"env": "prod"}, negate=True)
```

### DataAssetRule

데이터 자산 이름 패턴에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import DataAssetRule

# glob 패턴
rule = DataAssetRule(pattern="sales_*")

# 정규식
rule = DataAssetRule(pattern=r"^prod_.*_v\d+$", is_regex=True)

# 대소문자 무시
rule = DataAssetRule(pattern="USERS*", case_sensitive=False)
```

### MetadataRule

메타데이터 값에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import MetadataRule

# 단순 비교
rule = MetadataRule(key_path="region", expected_value="us-east-1")

# 중첩 경로
rule = MetadataRule(key_path="config.settings.mode", expected_value="production")

# 비교 연산자
rule = MetadataRule(key_path="priority", expected_value=5, comparator="gt")  # > 5
rule = MetadataRule(key_path="priority", expected_value=5, comparator="gte")  # >= 5
rule = MetadataRule(key_path="priority", expected_value=5, comparator="lt")   # < 5

# 포함 여부
rule = MetadataRule(key_path="owners", expected_value="data-team", comparator="contains")

# 정규식
rule = MetadataRule(key_path="name", expected_value=r"v\d+", comparator="regex")

# 존재 여부
rule = MetadataRule(key_path="special_flag", comparator="exists")
```

**comparator**: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, `regex`, `exists`

### TimeWindowRule

시간대에 따라 매칭합니다. 업무 시간/비업무 시간 구분에 유용합니다.

```python
from truthound.checkpoint.routing.rules import TimeWindowRule

# 업무 시간 (9시-17시, 평일)
rule = TimeWindowRule(
    start_time="09:00",
    end_time="17:00",
    days_of_week=[0, 1, 2, 3, 4],  # 월-금
)

# 비업무 시간 (자정 넘김)
rule = TimeWindowRule(
    start_time="17:00",
    end_time="09:00",  # 다음 날 9시까지
)

# 주말
rule = TimeWindowRule(
    days_of_week=[5, 6],  # 토-일
)

# 타임존 지정
rule = TimeWindowRule(
    start_time="09:00",
    end_time="17:00",
    timezone="America/New_York",
)
```

### PassRateRule

통과율에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import PassRateRule

# 통과율 90% 미만
rule = PassRateRule(max_rate=90.0)

# 통과율 50-80% 사이
rule = PassRateRule(min_rate=50.0, max_rate=80.0)

# 통과율 95% 이상
rule = PassRateRule(min_rate=95.0)
```

### ErrorRule

에러 발생 여부 또는 패턴에 따라 매칭합니다.

```python
from truthound.checkpoint.routing.rules import ErrorRule

# 에러가 있으면
rule = ErrorRule()

# 타임아웃 에러
rule = ErrorRule(pattern=r"timeout|timed out")

# 에러가 없으면
rule = ErrorRule(negate=True)
```

---

## 조합 규칙 (Combinators)

### AllOf (AND)

모든 규칙이 매칭되어야 합니다.

```python
from truthound.checkpoint.routing import AllOf
from truthound.checkpoint.routing.rules import SeverityRule, TagRule

# Critical 이슈 AND 프로덕션 환경
rule = AllOf([
    SeverityRule(min_severity="critical"),
    TagRule(tags={"env": "prod"}),
])
```

### AnyOf (OR)

하나 이상의 규칙이 매칭되면 됩니다.

```python
from truthound.checkpoint.routing import AnyOf

# Critical 이슈 OR 에러 발생
rule = AnyOf([
    SeverityRule(min_severity="critical"),
    ErrorRule(),
])
```

### NotRule (NOT)

규칙의 결과를 반전합니다.

```python
from truthound.checkpoint.routing import NotRule

# 프로덕션이 아닌 경우
rule = NotRule(TagRule(tags={"env": "prod"}))
```

### 복합 조합

```python
# (Critical OR Error) AND Production AND 업무시간
complex_rule = AllOf([
    AnyOf([
        SeverityRule(min_severity="critical"),
        ErrorRule(),
    ]),
    TagRule(tags={"env": "prod"}),
    TimeWindowRule(start_time="09:00", end_time="18:00"),
])
```

---

## 라우팅 모드

### FIRST_MATCH

첫 번째 매칭되는 라우트만 실행합니다.

```python
router = ActionRouter(mode=RouteMode.FIRST_MATCH)

# 우선순위 순서로 평가됨
router.add_route(Route(
    name="critical",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=100,  # 먼저 평가
))

router.add_route(Route(
    name="high",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(...)],
    priority=80,  # 다음으로 평가
))

# Critical 이슈가 있으면 PagerDuty만 호출됨
```

### ALL_MATCHES

매칭되는 모든 라우트를 실행합니다.

```python
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# 모든 매칭 라우트 실행
router.add_route(Route(
    name="always_store",
    rule=AlwaysRule(),
    actions=[StoreValidationResult(...)],
))

router.add_route(Route(
    name="failure_alert",
    rule=StatusRule(statuses=["failure"]),
    actions=[SlackNotification(...)],
))

# 실패 시: StoreValidationResult + SlackNotification 모두 실행
```

### PRIORITY_GROUP

가장 높은 우선순위 그룹의 모든 라우트를 실행합니다.

```python
router = ActionRouter(mode=RouteMode.PRIORITY_GROUP)

router.add_route(Route(
    name="critical_pagerduty",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(...)],
    priority=100,
))

router.add_route(Route(
    name="critical_slack",
    rule=SeverityRule(min_severity="critical"),
    actions=[SlackNotification(...)],
    priority=100,  # 같은 우선순위
))

router.add_route(Route(
    name="high_slack",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(...)],
    priority=80,
))

# Critical 이슈: priority=100인 두 라우트 모두 실행
# High 이슈만 있을 때: priority=80인 라우트만 실행
```

---

## 전체 예시

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.routing import (
    ActionRouter, Route, AllOf, AnyOf
)
from truthound.checkpoint.routing.base import RouteMode
from truthound.checkpoint.routing.rules import (
    SeverityRule, TagRule, TimeWindowRule, StatusRule, AlwaysRule
)
from truthound.checkpoint.actions import (
    StoreValidationResult, SlackNotification, PagerDutyAction, EmailNotification
)

# Router 생성
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

# 1. 항상 결과 저장
router.add_route(Route(
    name="always_store",
    rule=AlwaysRule(),
    actions=[StoreValidationResult(store_path="./results")],
    priority=0,
))

# 2. Critical + Production → PagerDuty
router.add_route(Route(
    name="critical_prod",
    rule=AllOf([
        SeverityRule(min_severity="critical"),
        TagRule(tags={"env": "prod"}),
    ]),
    actions=[
        PagerDutyAction(routing_key="${PAGERDUTY_KEY}"),
        SlackNotification(
            webhook_url="${SLACK_WEBHOOK}",
            mention_on_failure=["@oncall"],
        ),
    ],
    priority=100,
))

# 3. 업무 시간 외 Critical → Email만
router.add_route(Route(
    name="critical_offhours",
    rule=AllOf([
        SeverityRule(min_severity="critical"),
        TimeWindowRule(start_time="18:00", end_time="09:00"),  # 비업무시간
    ]),
    actions=[EmailNotification(
        to_addresses=["oncall@example.com"],
    )],
    priority=90,
))

# 4. High 이슈 → Slack
router.add_route(Route(
    name="high_alert",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(
        webhook_url="${SLACK_WEBHOOK}",
        channel="#data-quality",
    )],
    priority=80,
))

# Checkpoint에 Router 연결
checkpoint = Checkpoint(
    name="production_check",
    data_source="prod_data.parquet",
    validators=["null", "duplicate", "range"],
    router=router,
    tags={"env": "prod"},
)

result = checkpoint.run()
```

---

## YAML 설정 (RouteConfigParser)

```yaml
routes:
  - name: critical_prod
    priority: 100
    rule:
      type: all_of
      rules:
        - type: severity
          min_severity: critical
        - type: tag
          tags:
            env: prod
    actions:
      - type: pagerduty
        routing_key: ${PAGERDUTY_KEY}
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        mention_on_failure:
          - "@oncall"

  - name: high_alert
    priority: 80
    rule:
      type: severity
      min_severity: high
    actions:
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        channel: "#data-quality"

  - name: always_store
    priority: 0
    rule:
      type: always
    actions:
      - type: store_result
        store_path: ./results
```
