"""Telegram notification action.

This action sends notifications to Telegram chats via Bot API
when checkpoint validations complete.
"""

from __future__ import annotations

import json
import urllib.parse
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
class TelegramConfig(ActionConfig):
    """Configuration for Telegram notification action.

    Attributes:
        bot_token: Telegram bot token from BotFather.
        chat_id: Target chat/channel/group ID.
        parse_mode: Message parse mode ("HTML", "Markdown", "MarkdownV2").
        disable_notification: Send silently without notification sound.
        disable_web_page_preview: Disable link previews.
        include_details: Include detailed statistics in message.
        custom_message: Custom message template.
        reply_to_message_id: Optional message ID to reply to.
        message_thread_id: For forum groups, thread ID.
    """

    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    disable_notification: bool = False
    disable_web_page_preview: bool = True
    include_details: bool = True
    custom_message: str | None = None
    reply_to_message_id: int | None = None
    message_thread_id: int | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE


class TelegramNotification(BaseAction[TelegramConfig]):
    """Action to send Telegram notifications.

    Sends formatted messages to Telegram chats via Bot API
    with validation results and statistics.

    Example:
        >>> action = TelegramNotification(
        ...     bot_token="123456:ABC-DEF...",
        ...     chat_id="-1001234567890",
        ...     notify_on="failure",
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "telegram_notification"
    API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

    @classmethod
    def _default_config(cls) -> TelegramConfig:
        return TelegramConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send Telegram notification."""
        import urllib.request
        import urllib.error

        config = self._config

        if not config.bot_token:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No bot token configured",
                error="bot_token is required",
            )

        if not config.chat_id:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No chat ID configured",
                error="chat_id is required",
            )

        # Build message
        text = self._build_message(checkpoint_result)

        # Build request payload
        payload = {
            "chat_id": config.chat_id,
            "text": text,
            "parse_mode": config.parse_mode,
            "disable_notification": config.disable_notification,
            "disable_web_page_preview": config.disable_web_page_preview,
        }

        if config.reply_to_message_id:
            payload["reply_to_message_id"] = config.reply_to_message_id

        if config.message_thread_id:
            payload["message_thread_id"] = config.message_thread_id

        # Send to Telegram
        api_url = self.API_BASE.format(token=config.bot_token)

        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                api_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body)

            if response_data.get("ok"):
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Telegram notification sent",
                    details={
                        "message_id": response_data.get("result", {}).get("message_id"),
                        "chat_id": config.chat_id,
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.ERROR,
                    message="Telegram API error",
                    error=response_data.get("description", "Unknown error"),
                )

        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Telegram notification",
                error=str(e),
            )

    def _build_message(self, checkpoint_result: "CheckpointResult") -> str:
        """Build Telegram message text."""
        config = self._config
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        # Status emoji
        emoji_map = {
            "success": "",
            "failure": "",
            "error": "",
            "warning": "",
        }
        status_emoji = emoji_map.get(status, "")

        if config.custom_message:
            return config.custom_message.format(
                checkpoint=checkpoint_result.checkpoint_name,
                status=status.upper(),
                run_id=checkpoint_result.run_id,
                data_asset=checkpoint_result.data_asset,
                total_issues=stats.total_issues if stats else 0,
                emoji=status_emoji,
            )

        # Build message based on parse mode
        if config.parse_mode == "HTML":
            return self._build_html_message(checkpoint_result, stats, status_emoji)
        elif config.parse_mode in ("Markdown", "MarkdownV2"):
            return self._build_markdown_message(checkpoint_result, stats, status_emoji)
        else:
            return self._build_plain_message(checkpoint_result, stats, status_emoji)

    def _build_html_message(
        self,
        checkpoint_result: "CheckpointResult",
        stats: Any,
        status_emoji: str,
    ) -> str:
        """Build HTML formatted message."""
        config = self._config
        status = checkpoint_result.status.value

        lines = [
            f"{status_emoji} <b>Checkpoint: {self._escape_html(checkpoint_result.checkpoint_name)}</b>",
            "",
            f"<b>Status:</b> {status.upper()}",
        ]

        if config.include_details:
            lines.append(f"<b>Data Asset:</b> {self._escape_html(checkpoint_result.data_asset or 'N/A')}")
            lines.append(f"<b>Run ID:</b> <code>{checkpoint_result.run_id}</code>")

            if stats:
                lines.append(f"<b>Total Issues:</b> {stats.total_issues}")
                lines.append(f"<b>Pass Rate:</b> {stats.pass_rate * 100:.1f}%")

                if stats.total_issues > 0:
                    lines.append("")
                    lines.append("<b>Issues by Severity:</b>")
                    lines.append(f"  Critical: {stats.critical_issues}")
                    lines.append(f"  High: {stats.high_issues}")
                    lines.append(f"  Medium: {stats.medium_issues}")
                    lines.append(f"  Low: {stats.low_issues}")

            lines.append("")
            lines.append(f"<i>Run Time: {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}</i>")

        return "\n".join(lines)

    def _build_markdown_message(
        self,
        checkpoint_result: "CheckpointResult",
        stats: Any,
        status_emoji: str,
    ) -> str:
        """Build Markdown formatted message."""
        config = self._config
        status = checkpoint_result.status.value

        lines = [
            f"{status_emoji} *Checkpoint: {self._escape_markdown(checkpoint_result.checkpoint_name)}*",
            "",
            f"*Status:* {status.upper()}",
        ]

        if config.include_details:
            lines.append(f"*Data Asset:* {self._escape_markdown(checkpoint_result.data_asset or 'N/A')}")
            lines.append(f"*Run ID:* `{checkpoint_result.run_id}`")

            if stats:
                lines.append(f"*Total Issues:* {stats.total_issues}")
                lines.append(f"*Pass Rate:* {stats.pass_rate * 100:.1f}%")

                if stats.total_issues > 0:
                    lines.append("")
                    lines.append("*Issues by Severity:*")
                    lines.append(f"   Critical: {stats.critical_issues}")
                    lines.append(f"   High: {stats.high_issues}")
                    lines.append(f"   Medium: {stats.medium_issues}")
                    lines.append(f"   Low: {stats.low_issues}")

            lines.append("")
            lines.append(f"_Run Time: {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}_")

        return "\n".join(lines)

    def _build_plain_message(
        self,
        checkpoint_result: "CheckpointResult",
        stats: Any,
        status_emoji: str,
    ) -> str:
        """Build plain text message."""
        config = self._config
        status = checkpoint_result.status.value

        lines = [
            f"{status_emoji} Checkpoint: {checkpoint_result.checkpoint_name}",
            "",
            f"Status: {status.upper()}",
        ]

        if config.include_details:
            lines.append(f"Data Asset: {checkpoint_result.data_asset or 'N/A'}")
            lines.append(f"Run ID: {checkpoint_result.run_id}")

            if stats:
                lines.append(f"Total Issues: {stats.total_issues}")
                lines.append(f"Pass Rate: {stats.pass_rate * 100:.1f}%")

                if stats.total_issues > 0:
                    lines.append("")
                    lines.append("Issues by Severity:")
                    lines.append(f"  Critical: {stats.critical_issues}")
                    lines.append(f"  High: {stats.high_issues}")
                    lines.append(f"  Medium: {stats.medium_issues}")
                    lines.append(f"  Low: {stats.low_issues}")

            lines.append("")
            lines.append(f"Run Time: {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _escape_markdown(self, text: str) -> str:
        """Escape Markdown special characters."""
        # For MarkdownV2, more characters need escaping
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.bot_token:
            errors.append("bot_token is required")
        elif ":" not in self._config.bot_token:
            errors.append("bot_token appears to be invalid (should contain ':')")

        if not self._config.chat_id:
            errors.append("chat_id is required")

        if self._config.parse_mode not in ("HTML", "Markdown", "MarkdownV2", ""):
            errors.append(f"Invalid parse_mode: {self._config.parse_mode}")

        return errors


class TelegramNotificationWithPhoto(TelegramNotification):
    """Extended Telegram notification with photo support.

    Can send a photo along with the notification message.
    """

    action_type = "telegram_notification_photo"
    API_PHOTO = "https://api.telegram.org/bot{token}/sendPhoto"

    @dataclass
    class PhotoConfig(TelegramConfig):
        """Extended config with photo support."""
        photo_url: str | None = None
        caption_as_message: bool = True

    @classmethod
    def _default_config(cls) -> "TelegramNotificationWithPhoto.PhotoConfig":
        return TelegramNotificationWithPhoto.PhotoConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send Telegram notification with optional photo."""
        config = self._config

        # If no photo URL, fall back to regular message
        if not hasattr(config, "photo_url") or not config.photo_url:
            return super()._execute(checkpoint_result)

        import urllib.request
        import urllib.error

        text = self._build_message(checkpoint_result)

        payload = {
            "chat_id": config.chat_id,
            "photo": config.photo_url,
            "caption": text,
            "parse_mode": config.parse_mode,
        }

        api_url = self.API_PHOTO.format(token=config.bot_token)

        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                api_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body)

            if response_data.get("ok"):
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Telegram photo notification sent",
                    details={
                        "message_id": response_data.get("result", {}).get("message_id"),
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.ERROR,
                    message="Telegram API error",
                    error=response_data.get("description", "Unknown error"),
                )

        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Telegram photo notification",
                error=str(e),
            )
