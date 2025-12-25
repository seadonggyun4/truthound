"""Slack notification action.

This action sends notifications to Slack channels via webhooks
when checkpoint validations complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class SlackConfig(ActionConfig):
    """Configuration for Slack notification action.

    Attributes:
        webhook_url: Slack incoming webhook URL.
        channel: Optional channel override (if webhook supports it).
        username: Bot username to display.
        icon_emoji: Emoji icon for the bot.
        include_details: Include detailed statistics in message.
        mention_on_failure: User/group IDs to mention on failure.
        custom_message: Custom message template (supports placeholders).
    """

    webhook_url: str = ""
    channel: str | None = None
    username: str = "Truthound"
    icon_emoji: str = ":mag:"
    include_details: bool = True
    mention_on_failure: list[str] = field(default_factory=list)
    custom_message: str | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE


class SlackNotification(BaseAction[SlackConfig]):
    """Action to send Slack notifications.

    Sends formatted messages to Slack channels via incoming webhooks
    with validation results and statistics.

    Example:
        >>> action = SlackNotification(
        ...     webhook_url="https://hooks.slack.com/services/...",
        ...     notify_on="failure",
        ...     channel="#data-quality",
        ...     mention_on_failure=["U12345678"],
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "slack_notification"

    @classmethod
    def _default_config(cls) -> SlackConfig:
        return SlackConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send Slack notification."""
        import urllib.request
        import urllib.error

        config = self._config

        if not config.webhook_url:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No webhook URL configured",
                error="webhook_url is required",
            )

        # Build message payload
        payload = self._build_payload(checkpoint_result)

        # Send to Slack
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Slack notification sent",
                details={
                    "channel": config.channel,
                    "response": response_body,
                },
            )

        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Slack notification",
                error=str(e),
            )

    def _build_payload(self, checkpoint_result: "CheckpointResult") -> dict[str, Any]:
        """Build Slack message payload."""
        config = self._config
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        # Status colors
        color_map = {
            "success": "#28a745",
            "failure": "#dc3545",
            "error": "#dc3545",
            "warning": "#ffc107",
        }
        color = color_map.get(status, "#6c757d")

        # Status emoji
        emoji_map = {
            "success": ":white_check_mark:",
            "failure": ":x:",
            "error": ":exclamation:",
            "warning": ":warning:",
        }
        status_emoji = emoji_map.get(status, ":question:")

        # Build mentions for failure
        mentions = ""
        if status in ("failure", "error") and config.mention_on_failure:
            mentions = " ".join(f"<@{uid}>" for uid in config.mention_on_failure) + " "

        # Custom message or default
        if config.custom_message:
            text = config.custom_message.format(
                checkpoint=checkpoint_result.checkpoint_name,
                status=status.upper(),
                run_id=checkpoint_result.run_id,
                data_asset=checkpoint_result.data_asset,
                total_issues=stats.total_issues if stats else 0,
            )
        else:
            text = f"{mentions}{status_emoji} *Checkpoint '{checkpoint_result.checkpoint_name}'* completed with status *{status.upper()}*"

        # Build attachment with details
        attachment: dict[str, Any] = {
            "color": color,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                },
            ],
        }

        if config.include_details and stats:
            # Add statistics
            stats_block = {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Data Asset:*\n{checkpoint_result.data_asset}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n{checkpoint_result.run_id}"},
                    {"type": "mrkdwn", "text": f"*Total Issues:*\n{stats.total_issues}"},
                    {"type": "mrkdwn", "text": f"*Pass Rate:*\n{stats.pass_rate * 100:.1f}%"},
                ],
            }
            attachment["blocks"].append(stats_block)

            if stats.total_issues > 0:
                severity_block = {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f":red_circle: Critical: {stats.critical_issues} | "
                            f":large_orange_circle: High: {stats.high_issues} | "
                            f":large_yellow_circle: Medium: {stats.medium_issues} | "
                            f":large_blue_circle: Low: {stats.low_issues}",
                        }
                    ],
                }
                attachment["blocks"].append(severity_block)

        # Add timestamp
        attachment["blocks"].append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Run Time: {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}",
                }
            ],
        })

        payload: dict[str, Any] = {
            "username": config.username,
            "icon_emoji": config.icon_emoji,
            "attachments": [attachment],
        }

        if config.channel:
            payload["channel"] = config.channel

        return payload

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.webhook_url:
            errors.append("webhook_url is required")
        elif not self._config.webhook_url.startswith("https://hooks.slack.com/"):
            errors.append("webhook_url must be a valid Slack webhook URL")

        return errors
