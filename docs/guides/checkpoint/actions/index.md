# Actions 개요

Actions는 Checkpoint 검증이 완료된 후 실행되는 작업입니다. 결과 저장, 알림 발송, 외부 시스템 통합 등 다양한 용도로 사용됩니다.

## 액션 시스템 아키텍처

```
CheckpointResult
       │
       ▼
┌──────────────────────┐
│    Action Executor   │
│  (notify_on 조건)    │
└──────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│                   Actions                         │
├──────────┬──────────┬──────────┬────────────────┤
│ Storage  │Notify    │ Incident │ Integration    │
│          │          │          │                │
│ - Store  │ - Slack  │ - Pager  │ - Webhook      │
│ - Docs   │ - Email  │ - Ops    │ - GitHub       │
│          │ - Teams  │   Genie  │ - Custom       │
│          │ - Discord│          │                │
│          │ - Telegram          │                │
└──────────┴──────────┴──────────┴────────────────┘
```

## 액션 타입 비교표

| 액션 | 용도 | 주요 설정 | 기본 notify_on |
|------|------|-----------|----------------|
| **StoreValidationResult** | 결과 저장 | `store_path`, `partition_by`, `format` | `always` |
| **UpdateDataDocs** | HTML 리포트 생성 | `site_path`, `format`, `include_history` | `always` |
| **SlackNotification** | Slack 알림 | `webhook_url`, `channel`, `mention_on_failure` | `failure` |
| **EmailNotification** | 이메일 발송 | `smtp_host`, `to_addresses`, `provider` | `failure` |
| **TeamsNotification** | MS Teams 알림 | `webhook_url`, `card_builder` | `failure` |
| **DiscordNotification** | Discord 알림 | `webhook_url`, `embed_format` | `failure` |
| **TelegramNotification** | Telegram 알림 | `bot_token`, `chat_id` | `failure` |
| **PagerDutyAction** | 인시던트 생성 | `routing_key`, `auto_severity` | `failure_or_error` |
| **OpsGenieAction** | OpsGenie 알림 | `api_key`, `responders`, `priority` | `failure` |
| **WebhookAction** | HTTP 웹훅 | `url`, `method`, `auth_type` | `always` |
| **GitHubAction** | GitHub 통합 | `token`, `repo`, `check_name` | `always` |
| **CustomAction** | 사용자 정의 | `callback`, `shell_command` | `always` |

## 기본 액션 클래스

모든 액션은 `BaseAction[ConfigT]`를 상속합니다.

```python
# src/truthound/checkpoint/actions/base.py

class NotifyCondition(str, Enum):
    """액션 실행 조건."""
    ALWAYS = "always"              # 항상 실행
    SUCCESS = "success"            # 성공 시
    FAILURE = "failure"            # 실패 시
    ERROR = "error"                # 에러 시
    WARNING = "warning"            # 경고 시
    FAILURE_OR_ERROR = "failure_or_error"  # 실패 또는 에러
    NOT_SUCCESS = "not_success"    # 성공이 아닐 때


class ActionStatus(str, Enum):
    """액션 실행 결과 상태."""
    SUCCESS = "success"     # 성공
    FAILURE = "failure"     # 실패
    SKIPPED = "skipped"     # 건너뜀 (조건 미충족)
    ERROR = "error"         # 에러


@dataclass
class ActionConfig:
    """액션 기본 설정."""
    name: str | None = None            # 액션 이름 (생략 시 자동 생성)
    enabled: bool = True               # 활성화 여부
    notify_on: NotifyCondition = NotifyCondition.ALWAYS
    timeout_seconds: int = 30          # 타임아웃 (초)
    retry_count: int = 0               # 재시도 횟수
    retry_delay_seconds: float = 1.0   # 재시도 간격
    fail_checkpoint_on_error: bool = False  # 액션 실패 시 체크포인트 실패 처리
    metadata: dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class ActionResult:
    """액션 실행 결과."""
    action_name: str                   # 액션 이름
    action_type: str                   # 액션 타입
    status: ActionStatus               # 결과 상태
    message: str = ""                  # 결과 메시지
    error: str | None = None           # 에러 메시지
    details: dict[str, Any] = field(default_factory=dict)  # 상세 정보
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
```

## notify_on 조건

| 조건 | 실행 상황 |
|------|-----------|
| `always` | 모든 실행에서 |
| `success` | CheckpointStatus.SUCCESS |
| `failure` | CheckpointStatus.FAILURE |
| `error` | CheckpointStatus.ERROR |
| `warning` | CheckpointStatus.WARNING |
| `failure_or_error` | FAILURE 또는 ERROR |
| `not_success` | SUCCESS가 아닌 모든 경우 |

```python
# 예시: 조건별 액션 설정
actions = [
    # 항상 결과 저장
    StoreValidationResult(
        store_path="./results",
        notify_on="always",
    ),
    # 성공 시에만 문서 업데이트
    UpdateDataDocs(
        site_path="./docs",
        notify_on="success",
    ),
    # 실패 시 Slack 알림
    SlackNotification(
        webhook_url="...",
        notify_on="failure",
    ),
    # 실패 또는 에러 시 PagerDuty 호출
    PagerDutyAction(
        routing_key="...",
        notify_on="failure_or_error",
    ),
]
```

## 커스텀 액션 작성

`BaseAction`을 상속하여 커스텀 액션을 만들 수 있습니다.

```python
from dataclasses import dataclass
from truthound.checkpoint.actions.base import (
    BaseAction, ActionConfig, ActionResult, ActionStatus
)

@dataclass
class MyActionConfig(ActionConfig):
    """커스텀 액션 설정."""
    my_setting: str = ""
    another_setting: int = 0


class MyCustomAction(BaseAction[MyActionConfig]):
    """커스텀 액션."""

    action_type = "my_custom"

    @classmethod
    def _default_config(cls) -> MyActionConfig:
        return MyActionConfig()

    def _execute(self, checkpoint_result) -> ActionResult:
        """액션 실행 로직."""
        config = self._config

        # 여기에 커스텀 로직 구현
        try:
            # ... 작업 수행 ...
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Action completed",
                details={"my_setting": config.my_setting},
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Action failed",
                error=str(e),
            )

    def validate_config(self) -> list[str]:
        """설정 유효성 검사."""
        errors = []
        if not self._config.my_setting:
            errors.append("my_setting is required")
        return errors
```

## 하위 문서

- [Storage Actions](./storage.md) - 결과 저장 (StoreValidationResult, UpdateDataDocs)
- [Notification Actions](./notifications.md) - 알림 (Slack, Email, Teams, Discord, Telegram)
- [Incident Actions](./incident.md) - 인시던트 관리 (PagerDuty, OpsGenie)
- [Webhook Actions](./webhook.md) - HTTP 웹훅
- [Custom Actions](./custom.md) - 사용자 정의 액션
