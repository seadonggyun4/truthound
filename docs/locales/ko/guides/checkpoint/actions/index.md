# Actions 개요

실무 운영 가이드에서 Actions, They을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Action System 아키텍처

```
CheckpointResult
       │
       ▼
┌──────────────────────┐
│    Action Executor   │
│  (notify_on condition)│
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

## Action Type Comparison

| 실무 운영 가이드에서 Action을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Key, Settings을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|---------|--------------|-------------------|
| **Store검증Result** | Store 결과 | 실무 운영 가이드에서 `store_path`, `partition_by`, `format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 UpdateDataDocs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Generate HTML 리포트 | 실무 운영 가이드에서 `site_path`, `format`, `include_history`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 SlackNotification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Slack을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `webhook_url`, `channel`, `mention_on_failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 EmailNotification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Email을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `smtp_host`, `to_addresses`, `provider`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 TeamsNotification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Teams을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `webhook_url`, `card_builder`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 DiscordNotification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Discord을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `webhook_url`, `embed_format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 TelegramNotification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Telegram을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bot_token`, `chat_id`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 PagerDutyAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Incident을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `routing_key`, `auto_severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure_or_error`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 OpsGenieAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | OpsGenie 알림 | 실무 운영 가이드에서 `api_key`, `responders`, `priority`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 WebhookAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 HTTP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `url`, `method`, `auth_type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 GitHubAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | GitHub 통합 | 실무 운영 가이드에서 `token`, `repo`, `check_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CustomAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 User-defined을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `callback`, `shell_command`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Base Action Class

실무 운영 가이드에서 `BaseAction[ConfigT]`, BaseAction, ConfigT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
# src/truthound/checkpoint/actions/base.py

class NotifyCondition(str, Enum):
    """Action execution condition."""
    ALWAYS = "always"              # Always execute
    SUCCESS = "success"            # On success
    FAILURE = "failure"            # On failure
    ERROR = "error"                # On error
    WARNING = "warning"            # On warning
    FAILURE_OR_ERROR = "failure_or_error"  # On failure or error
    NOT_SUCCESS = "not_success"    # All cases except success


class ActionStatus(str, Enum):
    """Action execution result status."""
    SUCCESS = "success"     # Success
    FAILURE = "failure"     # Failure
    SKIPPED = "skipped"     # Skipped (condition not met)
    ERROR = "error"         # Error


@dataclass
class ActionConfig:
    """Base action configuration."""
    name: str | None = None            # Action name (auto-generated if omitted)
    enabled: bool = True               # Enable flag
    notify_on: NotifyCondition = NotifyCondition.ALWAYS
    timeout_seconds: int = 30          # Timeout in seconds
    retry_count: int = 0               # Retry count
    retry_delay_seconds: float = 1.0   # Retry interval
    fail_checkpoint_on_error: bool = False  # Fail checkpoint on action error
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class ActionResult:
    """Action execution result."""
    action_name: str                   # Action name
    action_type: str                   # Action type
    status: ActionStatus               # Result status
    message: str = ""                  # Result message
    error: str | None = None           # Error message
    details: dict[str, Any] = field(default_factory=dict)  # Detailed information
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
```

## notify_on Conditions

| 실무 운영 가이드에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution, Scenario을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------------|
| 실무 운영 가이드에서 `always`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `success`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트Status.SUCCESS |
| 실무 운영 가이드에서 `failure`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트Status.실패 |
| 실무 운영 가이드에서 `error`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트Status.ERROR |
| 실무 운영 가이드에서 `warning`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트Status.WARNING |
| 실무 운영 가이드에서 `failure_or_error`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실패 or ERROR |
| 실무 운영 가이드에서 `not_success`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SUCCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```python
# Example: Condition-based action configuration
actions = [
    # Always store results
    StoreValidationResult(
        store_path="./results",
        notify_on="always",
    ),
    # Update docs only on success
    UpdateDataDocs(
        site_path="./docs",
        notify_on="success",
    ),
    # Slack notification on failure
    SlackNotification(
        webhook_url="...",
        notify_on="failure",
    ),
    # PagerDuty call on failure or error
    PagerDutyAction(
        routing_key="...",
        notify_on="failure_or_error",
    ),
]
```

## Writing Custom Actions

실무 운영 가이드에서 `BaseAction`, Create, BaseAction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from dataclasses import dataclass
from truthound.checkpoint.actions.base import (
    BaseAction, ActionConfig, ActionResult, ActionStatus
)

@dataclass
class MyActionConfig(ActionConfig):
    """Custom action configuration."""
    my_setting: str = ""
    another_setting: int = 0


class MyCustomAction(BaseAction[MyActionConfig]):
    """Custom action."""

    action_type = "my_custom"

    @classmethod
    def _default_config(cls) -> MyActionConfig:
        return MyActionConfig()

    def _execute(self, checkpoint_result) -> ActionResult:
        """Action execution logic."""
        config = self._config

        # Implement custom logic here
        try:
            # ... perform work ...
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
        """Configuration validation."""
        errors = []
        if not self._config.my_setting:
            errors.append("my_setting is required")
        return errors
```

## Sub-documents

- 실무 운영 가이드에서 Storage, Actions, Result, StoreValidationResult, UpdateDataDocs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Notification, Actions, Notifications, Slack, Email, Teams, Discord, Telegram을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Incident, Actions, PagerDuty, OpsGenie을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Webhook, Actions, HTTP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Custom, Actions, User-defined을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
