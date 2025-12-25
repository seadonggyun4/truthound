"""Checkpoint actions for post-validation processing.

Actions are executed after validation completes. They can perform various
tasks like storing results, sending notifications, updating documentation,
or calling external webhooks.

Example:
    >>> from truthound.checkpoint.actions import (
    ...     StoreValidationResult,
    ...     SlackNotification,
    ...     WebhookAction,
    ... )
    >>>
    >>> actions = [
    ...     StoreValidationResult(store_path="./results"),
    ...     SlackNotification(webhook_url="...", notify_on="failure"),
    ...     WebhookAction(url="https://api.example.com/webhook"),
    ... ]
"""

from truthound.checkpoint.actions.base import (
    BaseAction,
    ActionConfig,
    ActionResult,
    ActionStatus,
    NotifyCondition,
)
from truthound.checkpoint.actions.store_result import (
    StoreValidationResult,
    StoreResultConfig,
)
from truthound.checkpoint.actions.update_docs import (
    UpdateDataDocs,
    UpdateDocsConfig,
)
from truthound.checkpoint.actions.slack_notify import (
    SlackNotification,
    SlackConfig,
)
from truthound.checkpoint.actions.email_notify import (
    EmailNotification,
    EmailConfig,
)
from truthound.checkpoint.actions.webhook import (
    WebhookAction,
    WebhookConfig,
)
from truthound.checkpoint.actions.pagerduty import (
    PagerDutyAction,
    PagerDutyConfig,
)
from truthound.checkpoint.actions.github_action import (
    GitHubAction,
    GitHubConfig,
)
from truthound.checkpoint.actions.custom import (
    CustomAction,
    CustomActionConfig,
)

__all__ = [
    # Base
    "BaseAction",
    "ActionConfig",
    "ActionResult",
    "ActionStatus",
    "NotifyCondition",
    # Store
    "StoreValidationResult",
    "StoreResultConfig",
    # Docs
    "UpdateDataDocs",
    "UpdateDocsConfig",
    # Notifications
    "SlackNotification",
    "SlackConfig",
    "EmailNotification",
    "EmailConfig",
    # Webhooks
    "WebhookAction",
    "WebhookConfig",
    # Integrations
    "PagerDutyAction",
    "PagerDutyConfig",
    "GitHubAction",
    "GitHubConfig",
    # Custom
    "CustomAction",
    "CustomActionConfig",
]
