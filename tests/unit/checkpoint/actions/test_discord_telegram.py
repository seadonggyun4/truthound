"""Tests for Discord and Telegram notification actions."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import json

from truthound.checkpoint.actions.discord_notify import (
    DiscordNotification,
    DiscordConfig,
)
from truthound.checkpoint.actions.telegram_notify import (
    TelegramNotification,
    TelegramConfig,
)
from truthound.checkpoint.actions.base import ActionStatus


class MockCheckpointResult:
    """Mock checkpoint result for testing."""

    def __init__(
        self,
        checkpoint_name: str = "test_checkpoint",
        status_value: str = "success",
        data_asset: str = "test_data",
        run_id: str = "run-123",
        total_issues: int = 0,
        pass_rate: float = 1.0,
    ):
        self.checkpoint_name = checkpoint_name
        self.data_asset = data_asset
        self.run_id = run_id
        self.run_time = datetime(2024, 1, 1, 10, 0, 0)

        self.status = Mock()
        self.status.value = status_value

        self.validation_result = Mock()
        self.validation_result.statistics = Mock()
        self.validation_result.statistics.total_issues = total_issues
        self.validation_result.statistics.critical_issues = 0
        self.validation_result.statistics.high_issues = 0
        self.validation_result.statistics.medium_issues = 0
        self.validation_result.statistics.low_issues = total_issues
        self.validation_result.statistics.pass_rate = pass_rate


class TestDiscordNotification:
    """Tests for DiscordNotification action."""

    def test_validate_config_missing_url(self) -> None:
        """Test validation with missing webhook URL."""
        action = DiscordNotification(name="test")
        errors = action.validate_config()

        assert len(errors) > 0
        assert any("webhook_url" in e for e in errors)

    def test_validate_config_invalid_url(self) -> None:
        """Test validation with invalid webhook URL."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://invalid.com/webhook",
        )
        errors = action.validate_config()

        assert len(errors) > 0
        assert any("webhook_url" in e.lower() for e in errors)

    def test_validate_config_valid_url(self) -> None:
        """Test validation with valid webhook URL."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )
        errors = action.validate_config()

        assert len(errors) == 0

    def test_build_payload_success(self) -> None:
        """Test building payload for success status."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="TestBot",
        )

        result = MockCheckpointResult(status_value="success")
        payload = action._build_payload(result)

        assert payload["username"] == "TestBot"
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["color"] == 0x28A745  # Green

    def test_build_payload_failure(self) -> None:
        """Test building payload for failure status."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            mention_role_ids=["role123"],
            mention_user_ids=["user456"],
        )

        result = MockCheckpointResult(
            status_value="failure",
            total_issues=5,
            pass_rate=0.9,
        )
        payload = action._build_payload(result)

        assert payload["embeds"][0]["color"] == 0xDC3545  # Red
        assert "content" in payload  # Should have mentions
        assert "<@&role123>" in payload["content"]
        assert "<@user456>" in payload["content"]

    def test_build_payload_with_details(self) -> None:
        """Test building payload with details included."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            include_details=True,
        )

        result = MockCheckpointResult(
            status_value="success",
            total_issues=3,
        )
        payload = action._build_payload(result)

        embed = payload["embeds"][0]
        assert "fields" in embed
        assert len(embed["fields"]) > 0

        field_names = [f["name"] for f in embed["fields"]]
        assert "Status" in field_names
        assert "Data Asset" in field_names

    def test_build_payload_without_details(self) -> None:
        """Test building payload without details."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            include_details=False,
        )

        result = MockCheckpointResult()
        payload = action._build_payload(result)

        embed = payload["embeds"][0]
        assert "fields" not in embed or len(embed["fields"]) == 0

    def test_custom_message(self) -> None:
        """Test custom message template."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            custom_message="Custom: {checkpoint} - {status}",
        )

        result = MockCheckpointResult(
            checkpoint_name="my_check",
            status_value="success",
        )
        payload = action._build_payload(result)

        assert "Custom: my_check - SUCCESS" in payload["embeds"][0]["description"]

    @patch("urllib.request.urlopen")
    def test_execute_success(self, mock_urlopen: Mock) -> None:
        """Test successful execution."""
        mock_response = MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.SUCCESS
        assert mock_urlopen.called

    def test_execute_no_webhook(self) -> None:
        """Test execution with no webhook URL."""
        action = DiscordNotification(name="test")

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.ERROR
        assert "webhook_url" in action_result.error


class TestTelegramNotification:
    """Tests for TelegramNotification action."""

    def test_validate_config_missing_token(self) -> None:
        """Test validation with missing bot token."""
        action = TelegramNotification(
            name="test",
            chat_id="123",
        )
        errors = action.validate_config()

        assert any("bot_token" in e for e in errors)

    def test_validate_config_missing_chat_id(self) -> None:
        """Test validation with missing chat ID."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
        )
        errors = action.validate_config()

        assert any("chat_id" in e for e in errors)

    def test_validate_config_invalid_token(self) -> None:
        """Test validation with invalid token format."""
        action = TelegramNotification(
            name="test",
            bot_token="invalid",  # No colon
            chat_id="123",
        )
        errors = action.validate_config()

        assert any("token" in e.lower() for e in errors)

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        action = TelegramNotification(
            name="test",
            bot_token="123456:ABC-DEF",
            chat_id="-1001234567890",
        )
        errors = action.validate_config()

        assert len(errors) == 0

    def test_build_html_message(self) -> None:
        """Test building HTML formatted message."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
            parse_mode="HTML",
            include_details=True,
        )

        result = MockCheckpointResult(
            checkpoint_name="test<script>",  # Should be escaped
            status_value="success",
            total_issues=5,
        )

        message = action._build_message(result)

        assert "<b>" in message  # HTML formatting
        assert "<script>" not in message  # Should be escaped
        assert "&lt;script&gt;" in message

    def test_build_markdown_message(self) -> None:
        """Test building Markdown formatted message."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
            parse_mode="Markdown",
        )

        result = MockCheckpointResult(status_value="failure")
        message = action._build_message(result)

        assert "*" in message  # Markdown formatting

    def test_build_plain_message(self) -> None:
        """Test building plain text message."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
            parse_mode="",
        )

        result = MockCheckpointResult()
        message = action._build_message(result)

        assert "Checkpoint:" in message
        assert "*" not in message  # No formatting
        assert "<b>" not in message

    def test_custom_message(self) -> None:
        """Test custom message template."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
            custom_message="{emoji} Check: {checkpoint} is {status}",
        )

        result = MockCheckpointResult(
            checkpoint_name="my_check",
            status_value="success",
        )

        message = action._build_message(result)

        assert "Check: my_check is SUCCESS" in message

    @patch("urllib.request.urlopen")
    def test_execute_success(self, mock_urlopen: Mock) -> None:
        """Test successful execution."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ok": True,
            "result": {"message_id": 123},
        }).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="456",
        )

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.SUCCESS
        assert action_result.details["message_id"] == 123

    @patch("urllib.request.urlopen")
    def test_execute_api_error(self, mock_urlopen: Mock) -> None:
        """Test execution with API error response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ok": False,
            "description": "Bad Request: chat not found",
        }).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="invalid",
        )

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.ERROR
        assert "chat not found" in action_result.error

    def test_execute_no_token(self) -> None:
        """Test execution with no bot token."""
        action = TelegramNotification(
            name="test",
            chat_id="123",
        )

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.ERROR
        assert "bot_token" in action_result.error

    def test_execute_no_chat_id(self) -> None:
        """Test execution with no chat ID."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
        )

        result = MockCheckpointResult()
        action_result = action._execute(result)

        assert action_result.status == ActionStatus.ERROR
        assert "chat_id" in action_result.error

    def test_escape_html(self) -> None:
        """Test HTML escaping."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
        )

        assert action._escape_html("<>&") == "&lt;&gt;&amp;"

    def test_escape_markdown(self) -> None:
        """Test Markdown escaping."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
        )

        escaped = action._escape_markdown("_*test*_")
        assert "\\_" in escaped
        assert "\\*" in escaped


class TestNotificationConditions:
    """Tests for notification conditions."""

    def test_discord_notify_on_failure(self) -> None:
        """Test Discord notification only on failure."""
        action = DiscordNotification(
            name="test",
            webhook_url="https://discord.com/api/webhooks/123/abc",
            notify_on="failure",
        )

        # Success should not trigger
        assert not action.should_run("success")

        # Failure should trigger
        assert action.should_run("failure")

    def test_telegram_notify_always(self) -> None:
        """Test Telegram notification always."""
        action = TelegramNotification(
            name="test",
            bot_token="123:ABC",
            chat_id="123",
            notify_on="always",
        )

        assert action.should_run("success")
        assert action.should_run("failure")
