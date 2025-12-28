"""Discord notification action.

This action sends notifications to Discord channels via webhooks
when checkpoint validations complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
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
class DiscordConfig(ActionConfig):
    """Configuration for Discord notification action.

    Attributes:
        webhook_url: Discord webhook URL.
        username: Bot username to display.
        avatar_url: Avatar URL for the bot.
        include_details: Include detailed statistics in message.
        mention_role_ids: Role IDs to mention on failure.
        mention_user_ids: User IDs to mention on failure.
        embed_color_success: Embed color for success (hex as int).
        embed_color_failure: Embed color for failure (hex as int).
        embed_color_warning: Embed color for warning (hex as int).
        custom_message: Custom message template.
        thread_name: Optional thread name (for forum channels).
    """

    webhook_url: str = ""
    username: str = "Truthound"
    avatar_url: str | None = None
    include_details: bool = True
    mention_role_ids: list[str] = field(default_factory=list)
    mention_user_ids: list[str] = field(default_factory=list)
    embed_color_success: int = 0x28A745  # Green
    embed_color_failure: int = 0xDC3545  # Red
    embed_color_warning: int = 0xFFC107  # Yellow
    custom_message: str | None = None
    thread_name: str | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE


class DiscordNotification(BaseAction[DiscordConfig]):
    """Action to send Discord notifications.

    Sends rich embeds to Discord channels via webhooks
    with validation results and statistics.

    Example:
        >>> action = DiscordNotification(
        ...     webhook_url="https://discord.com/api/webhooks/...",
        ...     notify_on="failure",
        ...     mention_role_ids=["123456789"],
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "discord_notification"

    @classmethod
    def _default_config(cls) -> DiscordConfig:
        return DiscordConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send Discord notification."""
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

        # Send to Discord
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
                message="Discord notification sent",
                details={
                    "response": response_body if response_body else "OK",
                },
            )

        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Discord notification",
                error=str(e),
            )

    def _build_payload(self, checkpoint_result: "CheckpointResult") -> dict[str, Any]:
        """Build Discord message payload."""
        config = self._config
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        # Determine embed color
        if status in ("failure", "error"):
            color = config.embed_color_failure
        elif status == "warning":
            color = config.embed_color_warning
        else:
            color = config.embed_color_success

        # Status emoji
        emoji_map = {
            "success": "",
            "failure": "",
            "error": "",
            "warning": "",
        }
        status_emoji = emoji_map.get(status, "")

        # Build mentions content
        mentions = []
        if status in ("failure", "error"):
            for role_id in config.mention_role_ids:
                mentions.append(f"<@&{role_id}>")
            for user_id in config.mention_user_ids:
                mentions.append(f"<@{user_id}>")

        content = " ".join(mentions) if mentions else None

        # Build embed
        embed: dict[str, Any] = {
            "title": f"{status_emoji} Checkpoint: {checkpoint_result.checkpoint_name}",
            "description": self._build_description(checkpoint_result, stats),
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Truthound Data Quality",
            },
        }

        # Add fields for details
        if config.include_details:
            fields = []

            fields.append({
                "name": "Status",
                "value": status.upper(),
                "inline": True,
            })

            fields.append({
                "name": "Data Asset",
                "value": checkpoint_result.data_asset or "N/A",
                "inline": True,
            })

            fields.append({
                "name": "Run ID",
                "value": f"`{checkpoint_result.run_id}`",
                "inline": True,
            })

            if stats:
                fields.append({
                    "name": "Total Issues",
                    "value": str(stats.total_issues),
                    "inline": True,
                })

                fields.append({
                    "name": "Pass Rate",
                    "value": f"{stats.pass_rate * 100:.1f}%",
                    "inline": True,
                })

                if stats.total_issues > 0:
                    severity_text = (
                        f" Critical: {stats.critical_issues}\n"
                        f" High: {stats.high_issues}\n"
                        f" Medium: {stats.medium_issues}\n"
                        f" Low: {stats.low_issues}"
                    )
                    fields.append({
                        "name": "Issues by Severity",
                        "value": severity_text,
                        "inline": False,
                    })

            fields.append({
                "name": "Run Time",
                "value": checkpoint_result.run_time.strftime("%Y-%m-%d %H:%M:%S"),
                "inline": True,
            })

            embed["fields"] = fields

        payload: dict[str, Any] = {
            "username": config.username,
            "embeds": [embed],
        }

        if content:
            payload["content"] = content

        if config.avatar_url:
            payload["avatar_url"] = config.avatar_url

        if config.thread_name:
            payload["thread_name"] = config.thread_name

        return payload

    def _build_description(
        self,
        checkpoint_result: "CheckpointResult",
        stats: Any,
    ) -> str:
        """Build embed description."""
        config = self._config
        status = checkpoint_result.status.value

        if config.custom_message:
            return config.custom_message.format(
                checkpoint=checkpoint_result.checkpoint_name,
                status=status.upper(),
                run_id=checkpoint_result.run_id,
                data_asset=checkpoint_result.data_asset,
                total_issues=stats.total_issues if stats else 0,
            )

        if status == "success":
            return "Checkpoint validation completed successfully."
        elif status == "failure":
            issues = stats.total_issues if stats else "unknown"
            return f"Checkpoint validation failed with {issues} issue(s)."
        elif status == "error":
            return "Checkpoint validation encountered an error."
        else:
            return f"Checkpoint validation completed with status: {status}"

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.webhook_url:
            errors.append("webhook_url is required")
        elif not self._config.webhook_url.startswith("https://discord.com/api/webhooks/"):
            # Also accept canary and ptb URLs
            valid_prefixes = [
                "https://discord.com/api/webhooks/",
                "https://canary.discord.com/api/webhooks/",
                "https://ptb.discord.com/api/webhooks/",
                "https://discordapp.com/api/webhooks/",
            ]
            if not any(self._config.webhook_url.startswith(p) for p in valid_prefixes):
                errors.append("webhook_url must be a valid Discord webhook URL")

        return errors
