"""Tests for Microsoft Teams notification action.

This module contains comprehensive tests for the Teams notification
functionality including Adaptive Card building, templates, and HTTP handling.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_checkpoint_result() -> MagicMock:
    """Create a mock CheckpointResult for testing."""
    from truthound.checkpoint.actions.base import ActionStatus

    # Create mock validation statistics
    mock_stats = MagicMock()
    mock_stats.total_issues = 5
    mock_stats.critical_issues = 1
    mock_stats.high_issues = 2
    mock_stats.medium_issues = 1
    mock_stats.low_issues = 1
    mock_stats.pass_rate = 0.85

    # Create mock validation result
    mock_validation = MagicMock()
    mock_validation.statistics = mock_stats
    mock_validation.results = [
        MagicMock(
            validator_name="NotNullValidator",
            column="email",
            message="Found 2 null values",
        ),
        MagicMock(
            validator_name="RangeValidator",
            column="age",
            message="Found 1 value out of range",
        ),
    ]

    # Create mock checkpoint result
    mock_result = MagicMock()
    mock_result.checkpoint_name = "test_checkpoint"
    mock_result.run_id = "run_abc123def456"
    mock_result.run_time = datetime(2025, 12, 27, 10, 30, 0)
    mock_result.duration_ms = 1234.56
    mock_result.data_asset = "customers.csv"
    mock_result.validation_result = mock_validation
    mock_result.status = MagicMock(value="failure")

    return mock_result


@pytest.fixture
def mock_success_result() -> MagicMock:
    """Create a mock successful CheckpointResult."""
    mock_stats = MagicMock()
    mock_stats.total_issues = 0
    mock_stats.critical_issues = 0
    mock_stats.high_issues = 0
    mock_stats.medium_issues = 0
    mock_stats.low_issues = 0
    mock_stats.pass_rate = 1.0

    mock_validation = MagicMock()
    mock_validation.statistics = mock_stats
    mock_validation.results = []

    mock_result = MagicMock()
    mock_result.checkpoint_name = "test_checkpoint"
    mock_result.run_id = "run_success123"
    mock_result.run_time = datetime(2025, 12, 27, 10, 30, 0)
    mock_result.duration_ms = 500.0
    mock_result.data_asset = "customers.csv"
    mock_result.validation_result = mock_validation
    mock_result.status = MagicMock(value="success")

    return mock_result


# =============================================================================
# Adaptive Card Builder Tests
# =============================================================================


class TestAdaptiveCardBuilder:
    """Tests for AdaptiveCardBuilder class."""

    def test_basic_card_build(self):
        """Test building a basic Adaptive Card."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_text_block("Hello World")

        card = builder.build()

        assert card["type"] == "AdaptiveCard"
        assert card["$schema"] == "http://adaptivecards.io/schemas/adaptive-card.json"
        assert card["version"] == "1.4"
        assert len(card["body"]) == 1
        assert card["body"][0]["type"] == "TextBlock"
        assert card["body"][0]["text"] == "Hello World"

    def test_version_setting(self):
        """Test setting card version."""
        from truthound.checkpoint.actions.teams_notify import (
            AdaptiveCardBuilder,
            AdaptiveCardVersion,
        )

        builder = AdaptiveCardBuilder(AdaptiveCardVersion.V1_5)
        card = builder.build()

        assert card["version"] == "1.5"

    def test_text_block_options(self):
        """Test text block with various options."""
        from truthound.checkpoint.actions.teams_notify import (
            AdaptiveCardBuilder,
            TextWeight,
            TextSize,
            TextColor,
        )

        builder = AdaptiveCardBuilder()
        builder.add_text_block(
            "Styled Text",
            weight=TextWeight.BOLDER,
            size=TextSize.LARGE,
            color=TextColor.ATTENTION,
            is_subtle=True,
            max_lines=2,
            horizontal_alignment="center",
        )

        card = builder.build()
        text_block = card["body"][0]

        assert text_block["text"] == "Styled Text"
        assert text_block["weight"] == "bolder"
        assert text_block["size"] == "large"
        assert text_block["color"] == "attention"
        assert text_block["isSubtle"] is True
        assert text_block["maxLines"] == 2
        assert text_block["horizontalAlignment"] == "center"

    def test_fact_set(self):
        """Test adding a fact set."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_fact_set([
            ("Name", "John Doe"),
            ("Status", "Active"),
            ("Role", "Admin"),
        ])

        card = builder.build()
        fact_set = card["body"][0]

        assert fact_set["type"] == "FactSet"
        assert len(fact_set["facts"]) == 3
        assert fact_set["facts"][0] == {"title": "Name", "value": "John Doe"}

    def test_container(self):
        """Test adding a container."""
        from truthound.checkpoint.actions.teams_notify import (
            AdaptiveCardBuilder,
            CardContainerStyle,
        )

        builder = AdaptiveCardBuilder()
        builder.add_container(
            [{"type": "TextBlock", "text": "Inside container"}],
            style=CardContainerStyle.EMPHASIS,
            bleed=True,
        )

        card = builder.build()
        container = card["body"][0]

        assert container["type"] == "Container"
        assert container["style"] == "emphasis"
        assert container["bleed"] is True

    def test_column_set(self):
        """Test adding a column set."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        columns = [
            {
                "type": "Column",
                "width": "auto",
                "items": [{"type": "TextBlock", "text": "Col 1"}],
            },
            {
                "type": "Column",
                "width": "stretch",
                "items": [{"type": "TextBlock", "text": "Col 2"}],
            },
        ]

        builder = AdaptiveCardBuilder()
        builder.add_column_set(columns)

        card = builder.build()
        column_set = card["body"][0]

        assert column_set["type"] == "ColumnSet"
        assert len(column_set["columns"]) == 2

    def test_action_open_url(self):
        """Test adding an OpenUrl action."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_action_open_url(
            "View Details",
            "https://example.com",
            style="positive",
            tooltip="Click to view",
        )

        card = builder.build()

        assert "actions" in card
        assert len(card["actions"]) == 1

        action = card["actions"][0]
        assert action["type"] == "Action.OpenUrl"
        assert action["title"] == "View Details"
        assert action["url"] == "https://example.com"
        assert action["style"] == "positive"
        assert action["tooltip"] == "Click to view"

    def test_action_submit(self):
        """Test adding a Submit action."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_action_submit("Submit", data={"key": "value"})

        card = builder.build()
        action = card["actions"][0]

        assert action["type"] == "Action.Submit"
        assert action["data"] == {"key": "value"}

    def test_action_show_card(self):
        """Test adding a ShowCard action."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        nested_card = {"type": "AdaptiveCard", "body": []}

        builder = AdaptiveCardBuilder()
        builder.add_action_show_card("Show More", nested_card)

        card = builder.build()
        action = card["actions"][0]

        assert action["type"] == "Action.ShowCard"
        assert action["card"] == nested_card

    def test_image(self):
        """Test adding an image."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_image(
            "https://example.com/image.png",
            alt_text="Example Image",
            size="medium",
            style="person",
        )

        card = builder.build()
        image = card["body"][0]

        assert image["type"] == "Image"
        assert image["url"] == "https://example.com/image.png"
        assert image["altText"] == "Example Image"
        assert image["size"] == "medium"
        assert image["style"] == "person"

    def test_full_width(self):
        """Test enabling full width."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.enable_full_width()

        card = builder.build()

        assert card["msteams"]["width"] == "Full"

    def test_fallback_text(self):
        """Test setting fallback text."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.set_fallback_text("Fallback message for old clients")

        card = builder.build()

        assert card["fallbackText"] == "Fallback message for old clients"

    def test_build_message_card(self):
        """Test building a complete message card."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        builder = AdaptiveCardBuilder()
        builder.add_text_block("Test message")

        message = builder.build_message_card()

        assert message["type"] == "message"
        assert len(message["attachments"]) == 1
        assert message["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
        assert message["attachments"][0]["content"]["type"] == "AdaptiveCard"

    def test_fluent_api(self):
        """Test that builder supports fluent API."""
        from truthound.checkpoint.actions.teams_notify import AdaptiveCardBuilder

        card = (
            AdaptiveCardBuilder()
            .set_version("1.4")
            .add_text_block("Title", weight="bolder")
            .add_text_block("Subtitle", is_subtle=True)
            .add_fact_set([("Key", "Value")])
            .add_action_open_url("Click", "https://example.com")
            .build()
        )

        assert len(card["body"]) == 3
        assert len(card["actions"]) == 1


# =============================================================================
# Message Template Tests
# =============================================================================


class TestMessageTemplates:
    """Tests for message template system."""

    def test_get_template_default(self):
        """Test getting default template."""
        from truthound.checkpoint.actions.teams_notify import (
            get_template,
            DefaultTemplate,
            MessageTheme,
        )

        template = get_template(MessageTheme.DEFAULT)
        assert isinstance(template, DefaultTemplate)

        template = get_template("default")
        assert isinstance(template, DefaultTemplate)

    def test_get_template_minimal(self):
        """Test getting minimal template."""
        from truthound.checkpoint.actions.teams_notify import (
            get_template,
            MinimalTemplate,
            MessageTheme,
        )

        template = get_template(MessageTheme.MINIMAL)
        assert isinstance(template, MinimalTemplate)

    def test_get_template_detailed(self):
        """Test getting detailed template."""
        from truthound.checkpoint.actions.teams_notify import (
            get_template,
            DetailedTemplate,
            MessageTheme,
        )

        template = get_template(MessageTheme.DETAILED)
        assert isinstance(template, DetailedTemplate)

    def test_get_template_compact(self):
        """Test getting compact template."""
        from truthound.checkpoint.actions.teams_notify import (
            get_template,
            CompactTemplate,
            MessageTheme,
        )

        template = get_template(MessageTheme.COMPACT)
        assert isinstance(template, CompactTemplate)

    def test_get_template_invalid(self):
        """Test getting invalid template raises error."""
        from truthound.checkpoint.actions.teams_notify import get_template

        with pytest.raises(ValueError, match="Unknown theme"):
            get_template("nonexistent")

    def test_register_custom_template(self):
        """Test registering a custom template."""
        from truthound.checkpoint.actions.teams_notify import (
            register_template,
            get_template,
            MessageTemplate,
        )

        class CustomTemplate(MessageTemplate):
            def render(self, checkpoint_result, config):
                return {"type": "message", "custom": True}

        register_template("my_custom", CustomTemplate)
        template = get_template("my_custom")

        assert isinstance(template, CustomTemplate)

    def test_default_template_render(self, mock_checkpoint_result):
        """Test rendering default template."""
        from truthound.checkpoint.actions.teams_notify import (
            DefaultTemplate,
            TeamsConfig,
        )

        template = DefaultTemplate()
        config = TeamsConfig(
            webhook_url="https://test.webhook.office.com/test",
            include_details=True,
            include_actions=False,
        )

        message = template.render(mock_checkpoint_result, config)

        assert message["type"] == "message"
        assert len(message["attachments"]) == 1

        content = message["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert len(content["body"]) > 0

    def test_minimal_template_render(self, mock_checkpoint_result):
        """Test rendering minimal template."""
        from truthound.checkpoint.actions.teams_notify import (
            MinimalTemplate,
            TeamsConfig,
        )

        template = MinimalTemplate()
        config = TeamsConfig(webhook_url="https://test.webhook.office.com/test")

        message = template.render(mock_checkpoint_result, config)

        assert message["type"] == "message"
        content = message["attachments"][0]["content"]
        # Minimal template should have fewer body elements
        assert len(content["body"]) <= 2

    def test_compact_template_render(self, mock_checkpoint_result):
        """Test rendering compact template."""
        from truthound.checkpoint.actions.teams_notify import (
            CompactTemplate,
            TeamsConfig,
        )

        template = CompactTemplate()
        config = TeamsConfig(webhook_url="https://test.webhook.office.com/test")

        message = template.render(mock_checkpoint_result, config)

        assert message["type"] == "message"
        content = message["attachments"][0]["content"]
        # Compact uses column set
        assert content["body"][0]["type"] == "ColumnSet"

    def test_template_with_mentions(self, mock_checkpoint_result):
        """Test template with @mentions on failure."""
        from truthound.checkpoint.actions.teams_notify import (
            DefaultTemplate,
            TeamsConfig,
        )

        template = DefaultTemplate()
        config = TeamsConfig(
            webhook_url="https://test.webhook.office.com/test",
            mention_on_failure=[
                {"id": "user@example.com", "name": "John Doe"},
                "another@example.com",
            ],
        )

        message = template.render(mock_checkpoint_result, config)
        content = message["attachments"][0]["content"]

        # Check that mentions entities were added
        assert "msteams" in content
        assert "entities" in content["msteams"]
        assert len(content["msteams"]["entities"]) == 2

    def test_template_with_action_buttons(self, mock_checkpoint_result):
        """Test template with action buttons."""
        from truthound.checkpoint.actions.teams_notify import (
            DefaultTemplate,
            TeamsConfig,
        )

        template = DefaultTemplate()
        config = TeamsConfig(
            webhook_url="https://test.webhook.office.com/test",
            include_actions=True,
            dashboard_url="https://dashboard.example.com/runs/{run_id}",
            details_url="https://details.example.com/{checkpoint}",
        )

        message = template.render(mock_checkpoint_result, config)
        content = message["attachments"][0]["content"]

        assert "actions" in content
        assert len(content["actions"]) == 2

        # Verify URL substitution
        assert "run_abc123def456" in content["actions"][0]["url"]


# =============================================================================
# TeamsConfig Tests
# =============================================================================


class TestTeamsConfig:
    """Tests for TeamsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from truthound.checkpoint.actions.teams_notify import (
            TeamsConfig,
            TeamsConnectorType,
            MessageTheme,
            AdaptiveCardVersion,
            NotifyCondition,
        )

        config = TeamsConfig()

        assert config.webhook_url == ""
        assert config.connector_type == TeamsConnectorType.INCOMING_WEBHOOK
        assert config.theme == MessageTheme.DEFAULT
        assert config.card_version == AdaptiveCardVersion.V1_4
        assert config.full_width is True
        assert config.include_details is True
        assert config.include_actions is True
        assert config.mention_on_failure == []
        assert config.notify_on == NotifyCondition.FAILURE

    def test_config_with_string_enums(self):
        """Test config converts string enums properly."""
        from truthound.checkpoint.actions.teams_notify import (
            TeamsConfig,
            TeamsConnectorType,
            AdaptiveCardVersion,
        )

        config = TeamsConfig(
            webhook_url="https://test.webhook.office.com/test",
            connector_type="incoming_webhook",
            card_version="1.5",
        )

        assert config.connector_type == TeamsConnectorType.INCOMING_WEBHOOK
        assert config.card_version == AdaptiveCardVersion.V1_5

    def test_config_notify_on_conversion(self):
        """Test notify_on string conversion."""
        from truthound.checkpoint.actions.teams_notify import TeamsConfig
        from truthound.checkpoint.actions.base import NotifyCondition

        config = TeamsConfig(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="failure_or_error",
        )

        assert config.notify_on == NotifyCondition.FAILURE_OR_ERROR


# =============================================================================
# TeamsNotification Action Tests
# =============================================================================


class TestTeamsNotification:
    """Tests for TeamsNotification action class."""

    def test_default_config(self):
        """Test action has correct default config."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification

        action = TeamsNotification()
        assert action.action_type == "teams_notification"
        assert action.config.webhook_url == ""

    def test_config_via_kwargs(self):
        """Test config can be set via kwargs."""
        from truthound.checkpoint.actions.teams_notify import (
            TeamsNotification,
            MessageTheme,
        )

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            theme=MessageTheme.MINIMAL,
            include_details=False,
        )

        assert action.config.webhook_url == "https://test.webhook.office.com/test"
        assert action.config.theme == MessageTheme.MINIMAL
        assert action.config.include_details is False

    def test_validate_config_missing_url(self):
        """Test validation fails without webhook URL."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification

        action = TeamsNotification()
        errors = action.validate_config()

        assert len(errors) == 1
        assert "webhook_url is required" in errors[0]

    def test_validate_config_invalid_url(self):
        """Test validation fails with invalid URL."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification

        action = TeamsNotification(webhook_url="https://invalid.example.com/webhook")
        errors = action.validate_config()

        assert len(errors) == 1
        assert "does not appear to be a valid Teams webhook URL" in errors[0]

    def test_validate_config_valid_webhook_url(self):
        """Test validation passes with valid webhook URL."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification

        # Test various valid URL patterns
        valid_urls = [
            "https://example.webhook.office.com/webhookb2/abc",
            "https://outlook.office.com/webhook/abc123",
            "https://prod-01.westus.logic.azure.com/workflows/abc",
        ]

        for url in valid_urls:
            action = TeamsNotification(webhook_url=url)
            errors = action.validate_config()
            assert errors == [], f"URL {url} should be valid"

    def test_execute_without_url(self, mock_checkpoint_result):
        """Test execute returns error without URL."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        action = TeamsNotification()
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "No webhook URL configured" in result.message

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_execute_success(self, mock_client_class, mock_checkpoint_result):
        """Test successful execution."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        # Setup mock
        mock_client = MagicMock()
        mock_client.post.return_value = (200, "1")
        mock_client_class.return_value = mock_client

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert "Teams notification sent" in result.message
        mock_client.post.assert_called_once()

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_execute_failure(self, mock_client_class, mock_checkpoint_result):
        """Test execution failure."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.return_value = (400, "Bad Request")
        mock_client_class.return_value = mock_client

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "400" in result.message

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_execute_exception(self, mock_client_class, mock_checkpoint_result):
        """Test execution with exception."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "Network error" in result.error

    def test_execute_skipped_on_success(self, mock_success_result):
        """Test execution is skipped on success when notify_on=failure."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="failure",
        )
        result = action.execute(mock_success_result)

        assert result.status == ActionStatus.SKIPPED

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_execute_with_custom_payload(self, mock_client_class, mock_checkpoint_result):
        """Test execution with custom payload."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.return_value = (200, "1")
        mock_client_class.return_value = mock_client

        custom_payload = {
            "type": "message",
            "text": "Checkpoint ${checkpoint} status: ${status}",
        }

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="always",
            custom_payload=custom_payload,
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS

        # Verify placeholder substitution
        call_args = mock_client.post.call_args
        payload = call_args[0][1]
        assert "test_checkpoint" in payload["text"]
        assert "failure" in payload["text"]

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_execute_with_custom_template(self, mock_client_class, mock_checkpoint_result):
        """Test execution with custom template."""
        from truthound.checkpoint.actions.teams_notify import (
            TeamsNotification,
            MessageTemplate,
        )
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.return_value = (200, "1")
        mock_client_class.return_value = mock_client

        class CustomTemplate(MessageTemplate):
            def render(self, checkpoint_result, config):
                return {
                    "type": "message",
                    "text": f"Custom: {checkpoint_result.checkpoint_name}",
                }

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="always",
            custom_template=CustomTemplate(),
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS

        call_args = mock_client.post.call_args
        payload = call_args[0][1]
        assert payload["text"] == "Custom: test_checkpoint"


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestTeamsHTTPClient:
    """Tests for TeamsHTTPClient class."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.build_opener")
    def test_post_success(self, mock_build_opener, mock_urlopen):
        """Test successful POST request."""
        from truthound.checkpoint.actions.teams_notify import TeamsHTTPClient

        # Setup mock response
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = b"1"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_response
        mock_build_opener.return_value = mock_opener

        client = TeamsHTTPClient(timeout=30)
        status, body = client.post(
            "https://test.webhook.office.com/test",
            {"type": "message", "text": "Test"},
        )

        assert status == 200
        assert body == "1"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.build_opener")
    def test_post_with_proxy(self, mock_build_opener, mock_urlopen):
        """Test POST request with proxy."""
        from truthound.checkpoint.actions.teams_notify import TeamsHTTPClient

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = b"1"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_response
        mock_build_opener.return_value = mock_opener

        client = TeamsHTTPClient(
            timeout=30,
            proxy="http://proxy.example.com:8080",
        )
        status, body = client.post(
            "https://test.webhook.office.com/test",
            {"type": "message"},
        )

        assert status == 200
        # Verify ProxyHandler was used
        mock_build_opener.assert_called()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_teams_notification(self):
        """Test create_teams_notification function."""
        from truthound.checkpoint.actions.teams_notify import (
            create_teams_notification,
            MessageTheme,
        )
        from truthound.checkpoint.actions.base import NotifyCondition

        action = create_teams_notification(
            "https://test.webhook.office.com/test",
            notify_on=NotifyCondition.FAILURE_OR_ERROR,
            theme=MessageTheme.DETAILED,
        )

        assert action.config.webhook_url == "https://test.webhook.office.com/test"
        assert action.config.notify_on == NotifyCondition.FAILURE_OR_ERROR
        assert action.config.theme == MessageTheme.DETAILED

    def test_create_failure_alert(self):
        """Test create_failure_alert function."""
        from truthound.checkpoint.actions.teams_notify import create_failure_alert
        from truthound.checkpoint.actions.base import NotifyCondition

        action = create_failure_alert(
            "https://test.webhook.office.com/test",
            mention_users=["user@example.com"],
            dashboard_url="https://dashboard.example.com",
        )

        assert action.config.notify_on == NotifyCondition.FAILURE_OR_ERROR
        assert action.config.mention_on_failure == ["user@example.com"]
        assert action.config.dashboard_url == "https://dashboard.example.com"
        assert action.config.include_actions is True

    def test_create_summary_notification(self):
        """Test create_summary_notification function."""
        from truthound.checkpoint.actions.teams_notify import (
            create_summary_notification,
            MessageTheme,
        )
        from truthound.checkpoint.actions.base import NotifyCondition

        action = create_summary_notification(
            "https://test.webhook.office.com/test",
            notify_on=NotifyCondition.ALWAYS,
        )

        assert action.config.theme == MessageTheme.COMPACT
        assert action.config.include_details is False
        assert action.config.include_actions is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestTeamsNotificationIntegration:
    """Integration tests for Teams notification flow."""

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_full_notification_flow(self, mock_client_class, mock_checkpoint_result):
        """Test complete notification flow with all options."""
        from truthound.checkpoint.actions.teams_notify import (
            TeamsNotification,
            MessageTheme,
        )
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.return_value = (200, "1")
        mock_client_class.return_value = mock_client

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="failure_or_error",
            theme=MessageTheme.DETAILED,
            full_width=True,
            include_details=True,
            include_actions=True,
            mention_on_failure=["user@example.com"],
            dashboard_url="https://dashboard.example.com/runs/{run_id}",
            details_url="https://details.example.com/{checkpoint}",
        )

        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS

        # Verify payload structure
        call_args = mock_client.post.call_args
        payload = call_args[0][1]

        assert payload["type"] == "message"
        assert len(payload["attachments"]) == 1

        content = payload["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert "msteams" in content
        assert content["msteams"].get("width") == "Full"

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_success_notification_skipped(self, mock_client_class, mock_success_result):
        """Test that success results are skipped with failure-only config."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="failure",
        )

        result = action.execute(mock_success_result)

        assert result.status == ActionStatus.SKIPPED
        mock_client_class.assert_not_called()

    @patch("truthound.checkpoint.actions.teams_notify.TeamsHTTPClient")
    def test_success_notification_sent(self, mock_client_class, mock_success_result):
        """Test that success results are sent with always config."""
        from truthound.checkpoint.actions.teams_notify import TeamsNotification
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.return_value = (200, "1")
        mock_client_class.return_value = mock_client

        action = TeamsNotification(
            webhook_url="https://test.webhook.office.com/test",
            notify_on="always",
        )

        result = action.execute(mock_success_result)

        assert result.status == ActionStatus.SUCCESS
        mock_client.post.assert_called_once()


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """Test all expected exports are accessible from __init__."""
        from truthound.checkpoint.actions import (
            TeamsNotification,
            TeamsConfig,
            AdaptiveCardBuilder,
            MessageTemplate,
            MessageTheme,
            TeamsConnectorType,
            create_teams_notification,
            create_failure_alert,
            create_summary_notification,
            register_template,
        )

        # Just verify imports work
        assert TeamsNotification is not None
        assert TeamsConfig is not None
        assert AdaptiveCardBuilder is not None
        assert MessageTemplate is not None
        assert MessageTheme is not None
        assert TeamsConnectorType is not None
        assert create_teams_notification is not None
        assert create_failure_alert is not None
        assert create_summary_notification is not None
        assert register_template is not None

    def test_teams_notification_in_all(self):
        """Test TeamsNotification is in __all__."""
        from truthound.checkpoint import actions

        assert "TeamsNotification" in actions.__all__
        assert "TeamsConfig" in actions.__all__
        assert "AdaptiveCardBuilder" in actions.__all__
