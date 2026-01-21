# Actions Overview

Actions are tasks executed after checkpoint validation completes. They serve various purposes including result storage, notification delivery, and external system integration.

## Action System Architecture

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

| Action | Purpose | Key Settings | Default notify_on |
|--------|---------|--------------|-------------------|
| **StoreValidationResult** | Store results | `store_path`, `partition_by`, `format` | `always` |
| **UpdateDataDocs** | Generate HTML reports | `site_path`, `format`, `include_history` | `always` |
| **SlackNotification** | Slack notifications | `webhook_url`, `channel`, `mention_on_failure` | `failure` |
| **EmailNotification** | Email delivery | `smtp_host`, `to_addresses`, `provider` | `failure` |
| **TeamsNotification** | MS Teams notifications | `webhook_url`, `card_builder` | `failure` |
| **DiscordNotification** | Discord notifications | `webhook_url`, `embed_format` | `failure` |
| **TelegramNotification** | Telegram notifications | `bot_token`, `chat_id` | `failure` |
| **PagerDutyAction** | Incident creation | `routing_key`, `auto_severity` | `failure_or_error` |
| **OpsGenieAction** | OpsGenie alerts | `api_key`, `responders`, `priority` | `failure` |
| **WebhookAction** | HTTP webhooks | `url`, `method`, `auth_type` | `always` |
| **GitHubAction** | GitHub integration | `token`, `repo`, `check_name` | `always` |
| **CustomAction** | User-defined | `callback`, `shell_command` | `always` |

## Base Action Class

All actions inherit from `BaseAction[ConfigT]`.

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

| Condition | Execution Scenario |
|-----------|-------------------|
| `always` | Every execution |
| `success` | CheckpointStatus.SUCCESS |
| `failure` | CheckpointStatus.FAILURE |
| `error` | CheckpointStatus.ERROR |
| `warning` | CheckpointStatus.WARNING |
| `failure_or_error` | FAILURE or ERROR |
| `not_success` | All cases except SUCCESS |

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

Create custom actions by inheriting from `BaseAction`.

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

- [Storage Actions](./storage.md) - Result storage (StoreValidationResult, UpdateDataDocs)
- [Notification Actions](./notifications.md) - Notifications (Slack, Email, Teams, Discord, Telegram)
- [Incident Actions](./incident.md) - Incident management (PagerDuty, OpsGenie)
- [Webhook Actions](./webhook.md) - HTTP webhooks
- [Custom Actions](./custom.md) - User-defined actions
