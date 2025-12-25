"""Example hook plugin for Slack notifications.

This plugin demonstrates how to use the hook system to send
notifications when validation events occur.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.request import Request, urlopen
from urllib.error import URLError

from truthound.plugins import (
    HookPlugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    HookType,
)

logger = logging.getLogger(__name__)


@dataclass
class SlackNotifierConfig(PluginConfig):
    """Configuration for Slack notifier plugin.

    Attributes:
        webhook_url: Slack webhook URL (required).
        channel: Channel to post to (optional, uses webhook default).
        username: Bot username (default: "Truthound").
        icon_emoji: Emoji icon (default: ":bar_chart:").
        notify_on_success: Whether to notify on successful validation.
        notify_on_failure: Whether to notify on validation failures.
        min_severity: Minimum severity to trigger notification.
        mention_on_critical: User/group to mention on critical issues.
    """

    webhook_url: str = ""
    channel: str | None = None
    username: str = "Truthound"
    icon_emoji: str = ":bar_chart:"
    notify_on_success: bool = False
    notify_on_failure: bool = True
    min_severity: str = "medium"
    mention_on_critical: str | None = None


class SlackNotifierPlugin(HookPlugin):
    """Plugin for sending Slack notifications on validation events.

    This plugin hooks into the validation lifecycle and sends
    Slack messages when issues are found.

    Example:
        >>> from truthound.plugins import PluginManager
        >>> config = SlackNotifierConfig(
        ...     webhook_url="https://hooks.slack.com/services/...",
        ...     notify_on_failure=True,
        ...     mention_on_critical="@oncall",
        ... )
        >>> manager = PluginManager()
        >>> plugin = SlackNotifierPlugin(config)
        >>> manager.load_from_class(type(plugin), config)
    """

    def __init__(self, config: SlackNotifierConfig | None = None):
        # Ensure we use SlackNotifierConfig
        if config is None:
            config = SlackNotifierConfig()
        elif not isinstance(config, SlackNotifierConfig):
            # Convert from generic PluginConfig
            config = SlackNotifierConfig(**{
                k: v for k, v in vars(config).items()
                if hasattr(SlackNotifierConfig, k)
            })
        super().__init__(config)
        self._config: SlackNotifierConfig = config

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="slack-notifier",
            version="1.0.0",
            plugin_type=PluginType.HOOK,
            description="Send Slack notifications on validation events",
            author="Truthound Team",
            tags=("notification", "slack", "alerting"),
        )

    def _get_plugin_name(self) -> str:
        return "slack-notifier"

    def _get_plugin_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return "Slack notifications for validation events"

    def get_hooks(self) -> dict[str, Callable]:
        """Return hooks to register."""
        return {
            HookType.AFTER_VALIDATION.value: self._on_validation_complete,
            HookType.ON_ERROR.value: self._on_error,
        }

    def _on_validation_complete(
        self,
        datasource: Any = None,
        result: Any = None,
        issues: list | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle validation completion."""
        if not self._config.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        issues = issues or []
        has_issues = len(issues) > 0

        # Check if we should notify
        if has_issues and not self._config.notify_on_failure:
            return
        if not has_issues and not self._config.notify_on_success:
            return

        # Filter by severity if needed
        if self._config.min_severity and has_issues:
            severity_order = ["low", "medium", "high", "critical"]
            min_idx = severity_order.index(self._config.min_severity.lower())
            issues = [
                i for i in issues
                if severity_order.index(i.severity.value.lower()) >= min_idx
            ]
            if not issues:
                return

        # Build message
        message = self._build_validation_message(datasource, issues)
        self._send_message(message)

    def _on_error(
        self,
        error: Exception | None = None,
        context: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle errors."""
        if not self._config.webhook_url:
            return

        message = {
            "text": ":x: Truthound Error",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {
                            "title": "Error",
                            "value": str(error) if error else "Unknown error",
                            "short": False,
                        },
                    ],
                }
            ],
        }

        if context:
            message["attachments"][0]["fields"].append({
                "title": "Context",
                "value": json.dumps(context, default=str)[:500],
                "short": False,
            })

        self._send_message(message)

    def _build_validation_message(
        self,
        datasource: Any,
        issues: list,
    ) -> dict[str, Any]:
        """Build Slack message for validation result."""
        source_name = getattr(datasource, "name", str(datasource)) if datasource else "Unknown"

        if not issues:
            return {
                "text": f":white_check_mark: Validation passed for `{source_name}`",
                "attachments": [
                    {
                        "color": "good",
                        "text": "No issues found.",
                    }
                ],
            }

        # Count by severity
        severity_counts = {}
        for issue in issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Determine color
        if severity_counts.get("critical", 0) > 0:
            color = "danger"
            emoji = ":rotating_light:"
        elif severity_counts.get("high", 0) > 0:
            color = "warning"
            emoji = ":warning:"
        else:
            color = "#ff9800"
            emoji = ":bell:"

        # Build text
        text = f"{emoji} Validation issues found in `{source_name}`"

        # Add mention for critical
        if (
            self._config.mention_on_critical
            and severity_counts.get("critical", 0) > 0
        ):
            text = f"{self._config.mention_on_critical} {text}"

        # Build fields
        fields = [
            {
                "title": "Total Issues",
                "value": str(len(issues)),
                "short": True,
            },
        ]

        for sev, count in sorted(severity_counts.items()):
            fields.append({
                "title": sev.capitalize(),
                "value": str(count),
                "short": True,
            })

        # Add top issues
        top_issues = issues[:5]
        issue_text = "\n".join([
            f"• `{i.column}`: {i.issue_type} ({i.severity.value})"
            for i in top_issues
        ])
        if len(issues) > 5:
            issue_text += f"\n... and {len(issues) - 5} more"

        return {
            "text": text,
            "attachments": [
                {
                    "color": color,
                    "fields": fields,
                    "text": issue_text,
                    "footer": "Truthound Data Quality",
                }
            ],
        }

    def _send_message(self, message: dict[str, Any]) -> bool:
        """Send message to Slack webhook."""
        if not self._config.webhook_url:
            return False

        # Add optional fields
        if self._config.channel:
            message["channel"] = self._config.channel
        if self._config.username:
            message["username"] = self._config.username
        if self._config.icon_emoji:
            message["icon_emoji"] = self._config.icon_emoji

        try:
            data = json.dumps(message).encode("utf-8")
            request = Request(
                self._config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=10) as response:
                return response.status == 200

        except URLError as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False

    def setup(self) -> None:
        """Validate configuration on setup."""
        if not self._config.webhook_url:
            logger.warning(
                "Slack notifier webhook URL not configured. "
                "Notifications will be disabled."
            )

    def teardown(self) -> None:
        """Cleanup."""
        pass

    def health_check(self) -> bool:
        """Check if plugin can send notifications."""
        if not self._config.webhook_url:
            return False

        # Could optionally test the webhook here
        return True
