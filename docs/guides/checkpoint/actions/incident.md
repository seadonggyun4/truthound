# Incident Actions

인시던트 관리 시스템과 통합하는 액션입니다. PagerDuty와 OpsGenie를 지원합니다.

## PagerDutyAction

PagerDuty Events API v2를 통해 인시던트를 생성하고 관리합니다.

### 설정 (PagerDutyConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `routing_key` | `str` | `""` | Events API v2 라우팅 키 (Integration Key) |
| `severity` | `str` | `"error"` | 기본 severity: `critical`, `error`, `warning`, `info` |
| `auto_severity` | `bool` | `True` | 검증 결과에 따라 자동 severity 매핑 |
| `component` | `str` | `"data-quality"` | 컴포넌트 이름 |
| `group` | `str` | `"truthound"` | 논리적 그룹 |
| `class_type` | `str` | `"validation"` | 인시던트 클래스/타입 |
| `custom_details` | `dict` | `{}` | 추가 커스텀 상세 정보 |
| `dedup_key_template` | `str` | `"{checkpoint}_{data_asset}"` | 중복 제거 키 템플릿 |
| `resolve_on_success` | `bool` | `True` | 성공 시 인시던트 자동 해결 |
| `api_endpoint` | `str` | `"https://events.pagerduty.com/v2/enqueue"` | API 엔드포인트 |
| `notify_on` | `str` | `"failure_or_error"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import PagerDutyAction

# 기본 사용
action = PagerDutyAction(
    routing_key="${PAGERDUTY_ROUTING_KEY}",
    auto_severity=True,
    resolve_on_success=True,
)

# 상세 설정
action = PagerDutyAction(
    routing_key="${PAGERDUTY_ROUTING_KEY}",
    severity="critical",  # auto_severity=False일 때 사용
    auto_severity=False,
    component="production-etl",
    group="data-platform",
    class_type="data-quality-validation",
    custom_details={
        "team": "data-engineering",
        "runbook": "https://wiki.example.com/dq-runbook",
    },
    resolve_on_success=True,
    notify_on="failure_or_error",
)

# 커스텀 중복 제거 키
action = PagerDutyAction(
    routing_key="...",
    dedup_key_template="{checkpoint}_{data_asset}_{run_id}",  # 실행별 별도 인시던트
)
```

### auto_severity 매핑

`auto_severity=True`일 때 검증 결과에 따른 PagerDuty severity 매핑:

| 조건 | PagerDuty Severity |
|------|---------------------|
| `critical_issues > 0` | `critical` |
| `high_issues > 0` | `error` |
| `medium_issues > 0` | `warning` |
| 그 외 | `info` |

### 인시던트 생명주기

1. **Trigger**: 검증 실패 시 인시던트 생성
2. **Dedup**: 동일 `dedup_key`로 중복 인시던트 방지
3. **Resolve**: `resolve_on_success=True`이고 성공 시 자동 해결

```python
# 예시 흐름
# 1. 첫 번째 실행 - 실패 → 인시던트 생성
# 2. 두 번째 실행 - 실패 → 기존 인시던트 업데이트 (dedup_key 동일)
# 3. 세 번째 실행 - 성공 → 인시던트 해결
```

### 전송 페이로드 예시

```json
{
  "routing_key": "xxx",
  "event_action": "trigger",
  "dedup_key": "daily_validation_users.csv",
  "payload": {
    "summary": "Data quality failure for 'daily_validation' on users.csv - 150 issues found (5 critical)",
    "severity": "critical",
    "source": "users.csv",
    "component": "data-quality",
    "group": "truthound",
    "class": "validation",
    "timestamp": "2024-01-15T12:00:00",
    "custom_details": {
      "checkpoint": "daily_validation",
      "run_id": "20240115_120000",
      "status": "failure",
      "data_asset": "users.csv",
      "statistics": {
        "total_issues": 150,
        "critical": 5,
        "high": 25,
        "medium": 70,
        "low": 50,
        "pass_rate": "85.0%"
      }
    }
  }
}
```

---

## OpsGenieAction

OpsGenie Alert API를 통해 알림을 생성하고 관리합니다.

### 설정 (OpsGenieConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `api_key` | `str` | `""` | OpsGenie API 키 |
| `region` | `str` | `"us"` | 리전: `us`, `eu` |
| `priority` | `AlertPriority` | `P3` | 기본 우선순위: `P1`-`P5` |
| `auto_priority` | `bool` | `True` | 검증 결과에 따라 자동 우선순위 매핑 |
| `responders` | `list[Responder]` | `[]` | 응답자 목록 |
| `visible_to` | `list[Responder]` | `[]` | 알림 공개 대상 |
| `tags` | `list[str]` | `[]` | 태그 목록 |
| `actions` | `list[str]` | `[]` | 액션 버튼 목록 |
| `alias_template` | `str` | `"{checkpoint}_{data_asset}"` | 알림 별칭 템플릿 (중복 제거용) |
| `auto_close_on_success` | `bool` | `True` | 성공 시 자동 종료 |
| `notify_on` | `str` | `"failure"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import OpsGenieAction
from truthound.checkpoint.actions.opsgenie import (
    AlertPriority,
    Responder,
    ResponderType,
)

# 기본 사용
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    auto_priority=True,
)

# 응답자 설정
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    responders=[
        Responder(type=ResponderType.TEAM, name="data-platform"),
        Responder(type=ResponderType.USER, username="oncall@example.com"),
        Responder(type=ResponderType.ESCALATION, name="data-quality-escalation"),
    ],
    visible_to=[
        Responder(type=ResponderType.TEAM, name="engineering"),
    ],
    priority=AlertPriority.P1,
    auto_priority=False,
    tags=["data-quality", "production", "automated"],
    auto_close_on_success=True,
)

# EU 리전
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    region="eu",  # api.eu.opsgenie.com 사용
)
```

### auto_priority 매핑

`auto_priority=True`일 때 검증 결과에 따른 OpsGenie 우선순위 매핑:

| 조건 | OpsGenie Priority |
|------|-------------------|
| `critical_issues > 0` | `P1` (Critical) |
| `high_issues > 0` | `P2` (High) |
| `medium_issues > 0` | `P3` (Moderate) |
| `low_issues > 0` | `P4` (Low) |
| 그 외 | `P5` (Informational) |

### Responder 타입

```python
class ResponderType(str, Enum):
    TEAM = "team"              # 팀
    USER = "user"              # 사용자 (username)
    ESCALATION = "escalation"  # 에스컬레이션 정책
    SCHEDULE = "schedule"      # 스케줄
```

### 팩토리 함수

```python
from truthound.checkpoint.actions.opsgenie import (
    create_opsgenie_action,
    create_critical_alert,
    create_team_alert,
)

# Critical 알림 생성
action = create_critical_alert(
    api_key="${OPSGENIE_API_KEY}",
    team="data-platform",
    escalation_policy="data-quality-escalation",
)

# 팀 알림 생성
action = create_team_alert(
    api_key="${OPSGENIE_API_KEY}",
    team="data-platform",
)
```

---

## YAML 설정 예시

```yaml
actions:
  # PagerDuty
  - type: pagerduty
    routing_key: ${PAGERDUTY_ROUTING_KEY}
    auto_severity: true
    resolve_on_success: true
    component: production-etl
    group: data-platform
    notify_on: failure_or_error

  # OpsGenie
  - type: opsgenie
    api_key: ${OPSGENIE_API_KEY}
    region: us
    auto_priority: true
    responders:
      - type: team
        name: data-platform
      - type: user
        username: oncall@example.com
    tags:
      - data-quality
      - production
    auto_close_on_success: true
    notify_on: failure
```

## 비교: PagerDuty vs OpsGenie

| 기능 | PagerDuty | OpsGenie |
|------|-----------|----------|
| **인시던트 생성** | Events API v2 | Alert API |
| **중복 제거** | `dedup_key` | `alias` |
| **자동 해결** | `event_action: resolve` | Close Alert API |
| **우선순위** | severity (4단계) | priority (P1-P5) |
| **응답자** | 에스컬레이션 정책 | 팀, 사용자, 스케줄, 정책 |
| **리전** | 단일 | US, EU |
