"""Tests for OpsGenie notification action.

This module contains comprehensive tests for the OpsGenie notification
functionality including alert payload building, templates, responders, and HTTP handling.
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
    # Create mock validation statistics
    mock_stats = MagicMock()
    mock_stats.total_issues = 5
    mock_stats.critical_issues = 1
    mock_stats.high_issues = 2
    mock_stats.medium_issues = 1
    mock_stats.low_issues = 1
    mock_stats.pass_rate = 0.85

    # Create mock validation result with failed results
    mock_failed_result1 = MagicMock()
    mock_failed_result1.validator_name = "NotNullValidator"
    mock_failed_result1.column = "email"
    mock_failed_result1.message = "Found 2 null values"
    mock_failed_result1.passed = False

    mock_failed_result2 = MagicMock()
    mock_failed_result2.validator_name = "RangeValidator"
    mock_failed_result2.column = "age"
    mock_failed_result2.message = "Found 1 value out of range"
    mock_failed_result2.passed = False

    mock_validation = MagicMock()
    mock_validation.statistics = mock_stats
    mock_validation.results = [mock_failed_result1, mock_failed_result2]

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


@pytest.fixture
def mock_warning_result() -> MagicMock:
    """Create a mock warning CheckpointResult."""
    mock_stats = MagicMock()
    mock_stats.total_issues = 2
    mock_stats.critical_issues = 0
    mock_stats.high_issues = 0
    mock_stats.medium_issues = 1
    mock_stats.low_issues = 1
    mock_stats.pass_rate = 0.95

    mock_validation = MagicMock()
    mock_validation.statistics = mock_stats
    mock_validation.results = []

    mock_result = MagicMock()
    mock_result.checkpoint_name = "test_checkpoint"
    mock_result.run_id = "run_warning123"
    mock_result.run_time = datetime(2025, 12, 27, 10, 30, 0)
    mock_result.duration_ms = 600.0
    mock_result.data_asset = "customers.csv"
    mock_result.validation_result = mock_validation
    mock_result.status = MagicMock(value="warning")

    return mock_result


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create a mock OpsGenie HTTP client."""
    client = MagicMock()
    client.post.return_value = {
        "result": "Request will be processed",
        "took": 0.302,
        "requestId": "43a29c5c-3dbf-4fa4-9c26-f4f71023e120",
    }
    return client


# =============================================================================
# Enum Tests
# =============================================================================


class TestOpsGenieRegion:
    """Tests for OpsGenieRegion enum."""

    def test_us_region_api_url(self):
        """Test US region API URL."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieRegion

        assert OpsGenieRegion.US.api_url == "https://api.opsgenie.com"

    def test_eu_region_api_url(self):
        """Test EU region API URL."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieRegion

        assert OpsGenieRegion.EU.api_url == "https://api.eu.opsgenie.com"

    def test_region_string_conversion(self):
        """Test region string conversion."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieRegion

        assert str(OpsGenieRegion.US) == "us"
        assert str(OpsGenieRegion.EU) == "eu"


class TestAlertPriority:
    """Tests for AlertPriority enum."""

    def test_priority_values(self):
        """Test priority values P1-P5."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert str(AlertPriority.P1) == "P1"
        assert str(AlertPriority.P2) == "P2"
        assert str(AlertPriority.P3) == "P3"
        assert str(AlertPriority.P4) == "P4"
        assert str(AlertPriority.P5) == "P5"

    def test_from_severity_critical(self):
        """Test severity mapping for critical."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("critical") == AlertPriority.P1

    def test_from_severity_high(self):
        """Test severity mapping for high."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("high") == AlertPriority.P2

    def test_from_severity_medium(self):
        """Test severity mapping for medium."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("medium") == AlertPriority.P3
        assert AlertPriority.from_severity("moderate") == AlertPriority.P3

    def test_from_severity_low(self):
        """Test severity mapping for low."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("low") == AlertPriority.P4

    def test_from_severity_info(self):
        """Test severity mapping for info."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("info") == AlertPriority.P5
        assert AlertPriority.from_severity("informational") == AlertPriority.P5

    def test_from_severity_unknown(self):
        """Test severity mapping for unknown returns default P3."""
        from truthound.checkpoint.actions.opsgenie import AlertPriority

        assert AlertPriority.from_severity("unknown") == AlertPriority.P3


# =============================================================================
# Responder Tests
# =============================================================================


class TestResponder:
    """Tests for Responder class."""

    def test_user_responder(self):
        """Test creating a user responder."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder.user("admin@example.com")

        assert responder.type == ResponderType.USER
        assert responder.username == "admin@example.com"
        assert responder.to_dict() == {
            "type": "user",
            "username": "admin@example.com",
        }

    def test_team_responder_by_name(self):
        """Test creating a team responder by name."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder.team("data-quality-team")

        assert responder.type == ResponderType.TEAM
        assert responder.name == "data-quality-team"
        assert responder.to_dict() == {
            "type": "team",
            "name": "data-quality-team",
        }

    def test_team_responder_by_id(self):
        """Test creating a team responder by ID."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder.team(id="4513b7ea-3b91-438f-b7e4-e3e54af9147c")

        assert responder.type == ResponderType.TEAM
        assert responder.id == "4513b7ea-3b91-438f-b7e4-e3e54af9147c"
        assert responder.to_dict() == {
            "type": "team",
            "id": "4513b7ea-3b91-438f-b7e4-e3e54af9147c",
        }

    def test_escalation_responder(self):
        """Test creating an escalation responder."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder.escalation("critical-escalation")

        assert responder.type == ResponderType.ESCALATION
        assert responder.name == "critical-escalation"
        assert responder.to_dict() == {
            "type": "escalation",
            "name": "critical-escalation",
        }

    def test_schedule_responder(self):
        """Test creating a schedule responder."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder.schedule("on-call-schedule")

        assert responder.type == ResponderType.SCHEDULE
        assert responder.name == "on-call-schedule"
        assert responder.to_dict() == {
            "type": "schedule",
            "name": "on-call-schedule",
        }

    def test_responder_from_string_type(self):
        """Test responder with string type is normalized."""
        from truthound.checkpoint.actions.opsgenie import Responder, ResponderType

        responder = Responder(type="team", name="my-team")

        assert responder.type == ResponderType.TEAM

    def test_responder_id_takes_precedence(self):
        """Test that ID takes precedence over name in to_dict."""
        from truthound.checkpoint.actions.opsgenie import Responder

        responder = Responder(
            type="team",
            id="team-id-123",
            name="team-name",
        )

        result = responder.to_dict()
        assert result["id"] == "team-id-123"
        assert "name" not in result


# =============================================================================
# Alert Payload Builder Tests
# =============================================================================


class TestAlertPayloadBuilder:
    """Tests for AlertPayloadBuilder class."""

    def test_basic_payload_build(self):
        """Test building a basic alert payload."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        builder = AlertPayloadBuilder()
        payload = builder.set_message("Test alert").build()

        assert payload["message"] == "Test alert"
        assert payload["priority"] == "P3"  # Default

    def test_message_required(self):
        """Test that message is required."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        builder = AlertPayloadBuilder()

        with pytest.raises(ValueError, match="Message is required"):
            builder.build()

    def test_message_truncation(self):
        """Test message truncation at 130 characters."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        long_message = "x" * 200
        builder = AlertPayloadBuilder()
        payload = builder.set_message(long_message).build()

        assert len(payload["message"]) == 130

    def test_full_payload(self):
        """Test building a complete payload."""
        from truthound.checkpoint.actions.opsgenie import (
            AlertPayloadBuilder,
            AlertPriority,
            Responder,
        )

        builder = AlertPayloadBuilder()
        payload = (
            builder.set_message("Test alert")
            .set_alias("test-alias")
            .set_description("Test description")
            .set_priority(AlertPriority.P1)
            .add_responder(Responder.team("test-team"))
            .add_tag("test-tag")
            .add_tags(["tag1", "tag2"])
            .set_details({"key1": "value1"})
            .add_detail("key2", "value2")
            .set_entity("test-entity")
            .set_source("test-source")
            .set_user("test-user")
            .set_note("test-note")
            .add_action("View Dashboard")
            .build()
        )

        assert payload["message"] == "Test alert"
        assert payload["alias"] == "test-alias"
        assert payload["description"] == "Test description"
        assert payload["priority"] == "P1"
        assert len(payload["responders"]) == 1
        assert payload["responders"][0]["name"] == "test-team"
        assert payload["tags"] == ["test-tag", "tag1", "tag2"]
        assert payload["details"] == {"key1": "value1", "key2": "value2"}
        assert payload["entity"] == "test-entity"
        assert payload["source"] == "test-source"
        assert payload["user"] == "test-user"
        assert payload["note"] == "test-note"
        assert payload["actions"] == ["View Dashboard"]

    def test_alias_truncation(self):
        """Test alias truncation at 512 characters."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        long_alias = "x" * 600
        builder = AlertPayloadBuilder()
        payload = builder.set_message("Test").set_alias(long_alias).build()

        assert len(payload["alias"]) == 512

    def test_description_truncation(self):
        """Test description truncation at 15000 characters."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        long_desc = "x" * 20000
        builder = AlertPayloadBuilder()
        payload = builder.set_message("Test").set_description(long_desc).build()

        assert len(payload["description"]) == 15000

    def test_priority_from_string(self):
        """Test setting priority from string."""
        from truthound.checkpoint.actions.opsgenie import AlertPayloadBuilder

        builder = AlertPayloadBuilder()
        payload = builder.set_message("Test").set_priority("P2").build()

        assert payload["priority"] == "P2"

    def test_multiple_responders(self):
        """Test adding multiple responders."""
        from truthound.checkpoint.actions.opsgenie import (
            AlertPayloadBuilder,
            Responder,
        )

        builder = AlertPayloadBuilder()
        payload = (
            builder.set_message("Test")
            .add_responder(Responder.team("team1"))
            .add_responders([
                Responder.user("user1@example.com"),
                Responder.user("user2@example.com"),
            ])
            .build()
        )

        assert len(payload["responders"]) == 3

    def test_visible_to(self):
        """Test adding visible_to responders."""
        from truthound.checkpoint.actions.opsgenie import (
            AlertPayloadBuilder,
            Responder,
        )

        builder = AlertPayloadBuilder()
        payload = (
            builder.set_message("Test")
            .add_visible_to(Responder.team("observer-team"))
            .build()
        )

        assert len(payload["visibleTo"]) == 1
        assert payload["visibleTo"][0]["name"] == "observer-team"


# =============================================================================
# Alert Template Tests
# =============================================================================


class TestDefaultAlertTemplate:
    """Tests for DefaultAlertTemplate."""

    def test_build_payload_failure(self, mock_checkpoint_result):
        """Test building payload for failure status."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(api_key="test-key")
        payload = template.build_payload(mock_checkpoint_result, config)

        assert "FAILURE" in payload["message"]
        assert "test_checkpoint" in payload["message"]
        assert payload["priority"] == "P1"  # auto_priority, has critical issues
        assert "truthound_test_checkpoint_customers.csv" in payload["alias"]
        assert "Total Issues: 5" in payload["description"]

    def test_auto_priority_critical(self, mock_checkpoint_result):
        """Test auto priority mapping for critical issues."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(api_key="test-key", auto_priority=True)
        payload = template.build_payload(mock_checkpoint_result, config)

        assert payload["priority"] == "P1"

    def test_auto_priority_disabled(self, mock_checkpoint_result):
        """Test disabled auto priority uses config priority."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
            AlertPriority,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(
            api_key="test-key",
            auto_priority=False,
            priority=AlertPriority.P4,
        )
        payload = template.build_payload(mock_checkpoint_result, config)

        assert payload["priority"] == "P4"

    def test_responders_from_config(self, mock_checkpoint_result):
        """Test responders are included from config."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
            Responder,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(
            api_key="test-key",
            responders=[Responder.team("test-team")],
        )
        payload = template.build_payload(mock_checkpoint_result, config)

        assert len(payload["responders"]) == 1
        assert payload["responders"][0]["name"] == "test-team"

    def test_dict_responders_converted(self, mock_checkpoint_result):
        """Test dict responders are converted to Responder objects."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(
            api_key="test-key",
            responders=[{"type": "team", "name": "dict-team"}],
        )
        payload = template.build_payload(mock_checkpoint_result, config)

        assert len(payload["responders"]) == 1
        assert payload["responders"][0]["name"] == "dict-team"

    def test_custom_details_included(self, mock_checkpoint_result):
        """Test custom details are included in payload."""
        from truthound.checkpoint.actions.opsgenie import (
            DefaultAlertTemplate,
            OpsGenieConfig,
        )

        template = DefaultAlertTemplate()
        config = OpsGenieConfig(
            api_key="test-key",
            custom_details={"environment": "production", "pipeline": "daily"},
        )
        payload = template.build_payload(mock_checkpoint_result, config)

        assert payload["details"]["environment"] == "production"
        assert payload["details"]["pipeline"] == "daily"


class TestMinimalAlertTemplate:
    """Tests for MinimalAlertTemplate."""

    def test_build_minimal_payload(self, mock_checkpoint_result):
        """Test building minimal payload."""
        from truthound.checkpoint.actions.opsgenie import (
            MinimalAlertTemplate,
            OpsGenieConfig,
        )

        template = MinimalAlertTemplate()
        config = OpsGenieConfig(api_key="test-key")
        payload = template.build_payload(mock_checkpoint_result, config)

        assert "FAILURE" in payload["message"]
        assert "5 issues" in payload["message"]
        assert "description" not in payload  # Minimal has no description
        assert payload["source"] == "truthound"


class TestDetailedAlertTemplate:
    """Tests for DetailedAlertTemplate."""

    def test_build_detailed_payload(self, mock_checkpoint_result):
        """Test building detailed payload."""
        from truthound.checkpoint.actions.opsgenie import (
            DetailedAlertTemplate,
            OpsGenieConfig,
        )

        template = DetailedAlertTemplate()
        config = OpsGenieConfig(api_key="test-key")
        payload = template.build_payload(mock_checkpoint_result, config)

        assert "Data Quality Alert" in payload["message"]
        assert "# Data Quality Validation Alert" in payload["description"]
        assert "## Overview" in payload["description"]
        assert "## Validation Statistics" in payload["description"]
        assert "## Failed Validations" in payload["description"]
        assert "## Recommended Actions" in payload["description"]
        assert "View Dashboard" in payload["actions"]
        assert "Acknowledge" in payload["actions"]
        assert "Escalate" in payload["actions"]


class TestTemplateRegistry:
    """Tests for template registry functions."""

    def test_get_default_template(self):
        """Test getting default template."""
        from truthound.checkpoint.actions.opsgenie import (
            get_template,
            DefaultAlertTemplate,
        )

        template = get_template("default")
        assert isinstance(template, DefaultAlertTemplate)

    def test_get_minimal_template(self):
        """Test getting minimal template."""
        from truthound.checkpoint.actions.opsgenie import (
            get_template,
            MinimalAlertTemplate,
        )

        template = get_template("minimal")
        assert isinstance(template, MinimalAlertTemplate)

    def test_get_detailed_template(self):
        """Test getting detailed template."""
        from truthound.checkpoint.actions.opsgenie import (
            get_template,
            DetailedAlertTemplate,
        )

        template = get_template("detailed")
        assert isinstance(template, DetailedAlertTemplate)

    def test_get_unknown_template(self):
        """Test getting unknown template raises error."""
        from truthound.checkpoint.actions.opsgenie import get_template

        with pytest.raises(ValueError, match="Unknown template"):
            get_template("unknown")

    def test_register_custom_template(self, mock_checkpoint_result):
        """Test registering custom template."""
        from truthound.checkpoint.actions.opsgenie import (
            AlertTemplate,
            OpsGenieConfig,
            register_template,
            get_template,
        )

        class CustomTemplate(AlertTemplate):
            def build_payload(self, checkpoint_result, config):
                return {"message": "Custom alert", "priority": "P5"}

        register_template("custom", CustomTemplate)
        template = get_template("custom")

        assert isinstance(template, CustomTemplate)
        config = OpsGenieConfig(api_key="test-key")
        payload = template.build_payload(mock_checkpoint_result, config)
        assert payload["message"] == "Custom alert"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestOpsGenieConfig:
    """Tests for OpsGenieConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieConfig,
            OpsGenieRegion,
            AlertPriority,
        )
        from truthound.checkpoint.actions.base import NotifyCondition

        config = OpsGenieConfig()

        assert config.api_key == ""
        assert config.region == OpsGenieRegion.US
        assert config.priority == AlertPriority.P3
        assert config.auto_priority is True
        assert config.responders == []
        assert "truthound" in config.tags
        assert config.close_on_success is True
        assert config.template == "default"
        assert config.notify_on == NotifyCondition.FAILURE_OR_ERROR

    def test_string_region_conversion(self):
        """Test string region is converted to enum."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieConfig,
            OpsGenieRegion,
        )

        config = OpsGenieConfig(region="eu")
        assert config.region == OpsGenieRegion.EU

    def test_string_priority_conversion(self):
        """Test string priority is converted to enum."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieConfig,
            AlertPriority,
        )

        config = OpsGenieConfig(priority="p1")
        assert config.priority == AlertPriority.P1

    def test_notify_on_conversion(self):
        """Test string notify_on is converted to enum."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieConfig
        from truthound.checkpoint.actions.base import NotifyCondition

        config = OpsGenieConfig(notify_on="always")
        assert config.notify_on == NotifyCondition.ALWAYS


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestOpsGenieHTTPClient:
    """Tests for OpsGenieHTTPClient."""

    def test_us_region_base_url(self):
        """Test US region base URL."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieHTTPClient,
            OpsGenieRegion,
        )

        client = OpsGenieHTTPClient(api_key="test-key", region=OpsGenieRegion.US)
        assert client.base_url == "https://api.opsgenie.com"

    def test_eu_region_base_url(self):
        """Test EU region base URL."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieHTTPClient,
            OpsGenieRegion,
        )

        client = OpsGenieHTTPClient(api_key="test-key", region=OpsGenieRegion.EU)
        assert client.base_url == "https://api.eu.opsgenie.com"

    @patch("urllib.request.build_opener")
    def test_post_success(self, mock_build_opener):
        """Test successful POST request."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieHTTPClient

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"result": "success", "requestId": "123"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_response
        mock_build_opener.return_value = mock_opener

        client = OpsGenieHTTPClient(api_key="test-key")
        result = client.post("/v2/alerts", {"message": "Test"})

        assert result["result"] == "success"
        assert result["requestId"] == "123"

    @patch("urllib.request.build_opener")
    def test_post_http_error(self, mock_build_opener):
        """Test POST request with HTTP error."""
        import urllib.error
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieHTTPClient,
            OpsGenieAPIError,
        )

        mock_opener = MagicMock()
        mock_error = urllib.error.HTTPError(
            url="https://api.opsgenie.com/v2/alerts",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"message": "Invalid API key"}')),
        )
        mock_opener.open.side_effect = mock_error
        mock_build_opener.return_value = mock_opener

        client = OpsGenieHTTPClient(api_key="invalid-key")

        with pytest.raises(OpsGenieAPIError) as exc_info:
            client.post("/v2/alerts", {"message": "Test"})

        assert exc_info.value.status_code == 401
        assert "Unauthorized" in str(exc_info.value)


# =============================================================================
# Action Tests
# =============================================================================


class TestOpsGenieAction:
    """Tests for OpsGenieAction class."""

    def test_action_type(self):
        """Test action type is set correctly."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction

        action = OpsGenieAction(api_key="test-key")
        assert action.action_type == "opsgenie"

    def test_default_config(self):
        """Test default configuration."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction

        action = OpsGenieAction()
        assert action.config.api_key == ""

    def test_config_via_kwargs(self):
        """Test configuration via kwargs."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieAction,
            AlertPriority,
            Responder,
        )

        action = OpsGenieAction(
            api_key="test-key",
            priority=AlertPriority.P1,
            responders=[Responder.team("test-team")],
        )

        assert action.config.api_key == "test-key"
        assert action.config.priority == AlertPriority.P1
        assert len(action.config.responders) == 1

    def test_missing_api_key_error(self, mock_checkpoint_result):
        """Test error when API key is missing."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        action = OpsGenieAction()
        result = action._execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "api_key is required" in result.error

    def test_create_alert(self, mock_checkpoint_result, mock_http_client):
        """Test creating an alert."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        action = OpsGenieAction(
            api_key="test-key",
            client=mock_http_client,
        )
        result = action._execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert result.details["action"] == "create"
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/v2/alerts"

    def test_close_alert_on_success(self, mock_success_result, mock_http_client):
        """Test closing alert on success."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        action = OpsGenieAction(
            api_key="test-key",
            close_on_success=True,
            client=mock_http_client,
        )
        result = action._execute(mock_success_result)

        assert result.status == ActionStatus.SUCCESS
        assert result.details["action"] == "close"
        call_args = mock_http_client.post.call_args
        assert "/close" in call_args[0][0]
        assert "identifierType=alias" in call_args[0][0]

    def test_acknowledge_alert_on_warning(self, mock_warning_result, mock_http_client):
        """Test acknowledging alert on warning."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        action = OpsGenieAction(
            api_key="test-key",
            acknowledge_on_warning=True,
            client=mock_http_client,
        )
        result = action._execute(mock_warning_result)

        assert result.status == ActionStatus.SUCCESS
        assert result.details["action"] == "acknowledge"
        call_args = mock_http_client.post.call_args
        assert "/acknowledge" in call_args[0][0]

    def test_api_error_handling(self, mock_checkpoint_result):
        """Test API error handling."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieAction,
            OpsGenieAPIError,
        )
        from truthound.checkpoint.actions.base import ActionStatus

        mock_client = MagicMock()
        mock_client.post.side_effect = OpsGenieAPIError(
            "API Error",
            status_code=400,
            response_body='{"message": "Bad request"}',
        )

        action = OpsGenieAction(api_key="test-key", client=mock_client)
        result = action._execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "OpsGenie API error" in result.message
        assert result.details["status_code"] == 400

    def test_custom_template(self, mock_checkpoint_result, mock_http_client):
        """Test using custom template."""
        from truthound.checkpoint.actions.opsgenie import (
            OpsGenieAction,
            AlertTemplate,
            OpsGenieConfig,
        )
        from truthound.checkpoint.actions.base import ActionStatus

        class MyTemplate(AlertTemplate):
            def build_payload(self, checkpoint_result, config):
                return {"message": "Custom message", "priority": "P5"}

        action = OpsGenieAction(
            api_key="test-key",
            custom_template=MyTemplate(),
            client=mock_http_client,
        )
        result = action._execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        call_args = mock_http_client.post.call_args
        payload = call_args[0][1]
        assert payload["message"] == "Custom message"
        assert payload["priority"] == "P5"

    def test_validate_config_missing_api_key(self):
        """Test config validation with missing API key."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction

        action = OpsGenieAction()
        errors = action.validate_config()

        assert "api_key is required" in errors

    def test_validate_config_invalid_priority(self):
        """Test config validation with invalid priority."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction, OpsGenieConfig

        config = OpsGenieConfig(api_key="test-key")
        config.priority = "INVALID"  # type: ignore
        action = OpsGenieAction(config=config)
        errors = action.validate_config()

        assert any("priority" in e.lower() for e in errors)

    def test_validate_config_invalid_region(self):
        """Test config validation with invalid region."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction, OpsGenieConfig

        config = OpsGenieConfig(api_key="test-key")
        config.region = "invalid"  # type: ignore
        action = OpsGenieAction(config=config)
        errors = action.validate_config()

        assert any("region" in e.lower() for e in errors)

    def test_validate_config_invalid_responder(self):
        """Test config validation with invalid responder."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction

        action = OpsGenieAction(
            api_key="test-key",
            responders=[{"name": "missing-type"}],
        )
        errors = action.validate_config()

        assert any("type" in e.lower() for e in errors)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_opsgenie_action(self):
        """Test create_opsgenie_action function."""
        from truthound.checkpoint.actions.opsgenie import (
            create_opsgenie_action,
            AlertPriority,
        )
        from truthound.checkpoint.actions.base import NotifyCondition

        action = create_opsgenie_action(
            api_key="test-key",
            notify_on=NotifyCondition.ALWAYS,
            priority=AlertPriority.P2,
            template="detailed",
        )

        assert action.config.api_key == "test-key"
        assert action.config.notify_on == NotifyCondition.ALWAYS
        assert action.config.priority == AlertPriority.P2
        assert action.config.template == "detailed"

    def test_create_critical_alert(self):
        """Test create_critical_alert function."""
        from truthound.checkpoint.actions.opsgenie import (
            create_critical_alert,
            AlertPriority,
            Responder,
        )
        from truthound.checkpoint.actions.base import NotifyCondition

        action = create_critical_alert(
            api_key="test-key",
            responders=[Responder.team("critical-team")],
            tags=["critical", "p0"],
        )

        assert action.config.api_key == "test-key"
        assert action.config.priority == AlertPriority.P1
        assert action.config.auto_priority is False
        assert action.config.template == "detailed"
        assert action.config.close_on_success is True
        assert len(action.config.responders) == 1
        assert "critical" in action.config.tags

    def test_create_team_alert(self):
        """Test create_team_alert function."""
        from truthound.checkpoint.actions.opsgenie import create_team_alert

        action = create_team_alert(
            api_key="test-key",
            team_name="data-quality-team",
        )

        assert action.config.api_key == "test-key"
        assert len(action.config.responders) == 1
        responder = action.config.responders[0]
        assert responder.name == "data-quality-team"

    def test_create_escalation_alert(self):
        """Test create_escalation_alert function."""
        from truthound.checkpoint.actions.opsgenie import (
            create_escalation_alert,
            AlertPriority,
        )

        action = create_escalation_alert(
            api_key="test-key",
            escalation_name="critical-escalation",
        )

        assert action.config.api_key == "test-key"
        assert action.config.priority == AlertPriority.P1
        assert len(action.config.responders) == 1
        responder = action.config.responders[0]
        assert responder.name == "critical-escalation"
        assert str(responder.type) == "escalation"


# =============================================================================
# Integration Tests
# =============================================================================


class TestOpsGenieActionIntegration:
    """Integration tests for OpsGenieAction with execute() method."""

    def test_execute_skips_on_condition(self, mock_success_result):
        """Test action is skipped when notify_on condition not met."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus, NotifyCondition

        action = OpsGenieAction(
            api_key="test-key",
            notify_on=NotifyCondition.FAILURE,  # Only on failure
            close_on_success=False,  # Don't close on success
        )
        result = action.execute(mock_success_result)

        assert result.status == ActionStatus.SKIPPED

    def test_execute_runs_on_failure(self, mock_checkpoint_result, mock_http_client):
        """Test action executes on failure condition."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus, NotifyCondition

        action = OpsGenieAction(
            api_key="test-key",
            notify_on=NotifyCondition.FAILURE,
            client=mock_http_client,
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert result.duration_ms > 0
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_execute_with_retry(self, mock_checkpoint_result):
        """Test action retries on failure when _execute raises exception.

        Note: The OpsGenieAction._execute() method catches exceptions and returns
        ActionResult with ERROR status. The BaseAction.execute() retry logic only
        triggers when _execute() raises an exception. This test verifies the behavior
        when using a mock that raises exceptions (simulating network-level failures
        before the try/except in _execute).
        """
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        # Create a subclass that raises exceptions to trigger retry
        class RetryTestAction(OpsGenieAction):
            call_count = 0

            def _execute(self, checkpoint_result):
                self.call_count += 1
                if self.call_count < 3:
                    raise RuntimeError("Temporary failure")
                return super()._execute(checkpoint_result)

        mock_client = MagicMock()
        mock_client.post.return_value = {"result": "success", "requestId": "123"}

        action = RetryTestAction(
            api_key="test-key",
            retry_count=2,
            retry_delay_seconds=0.01,
            client=mock_client,
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert action.call_count == 3

    def test_execute_fails_after_retries(self, mock_checkpoint_result):
        """Test action fails after exhausting retries when _execute raises."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        # Create a subclass that always raises exceptions
        class AlwaysFailAction(OpsGenieAction):
            call_count = 0

            def _execute(self, checkpoint_result):
                self.call_count += 1
                raise RuntimeError("Permanent failure")

        action = AlwaysFailAction(
            api_key="test-key",
            retry_count=2,
            retry_delay_seconds=0.01,
        )
        result = action.execute(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert action.call_count == 3  # Initial + 2 retries
        assert "Permanent failure" in result.error


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_data_asset(self, mock_checkpoint_result, mock_http_client):
        """Test handling of empty data asset."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        mock_checkpoint_result.data_asset = None

        action = OpsGenieAction(api_key="test-key", client=mock_http_client)
        result = action._execute(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS

    def test_no_validation_result(self, mock_http_client):
        """Test handling of missing validation result."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        mock_result = MagicMock()
        mock_result.checkpoint_name = "test"
        mock_result.run_id = "run123"
        mock_result.run_time = datetime.now()
        mock_result.data_asset = "test.csv"
        mock_result.validation_result = None
        mock_result.status = MagicMock(value="failure")

        action = OpsGenieAction(api_key="test-key", client=mock_http_client)
        result = action._execute(mock_result)

        assert result.status == ActionStatus.SUCCESS

    def test_no_statistics(self, mock_http_client):
        """Test handling of missing statistics."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        mock_validation = MagicMock()
        mock_validation.statistics = None
        mock_validation.results = []

        mock_result = MagicMock()
        mock_result.checkpoint_name = "test"
        mock_result.run_id = "run123"
        mock_result.run_time = datetime.now()
        mock_result.data_asset = "test.csv"
        mock_result.validation_result = mock_validation
        mock_result.status = MagicMock(value="failure")

        action = OpsGenieAction(api_key="test-key", client=mock_http_client)
        result = action._execute(mock_result)

        assert result.status == ActionStatus.SUCCESS

    def test_special_characters_in_message(self, mock_http_client):
        """Test handling of special characters."""
        from truthound.checkpoint.actions.opsgenie import OpsGenieAction
        from truthound.checkpoint.actions.base import ActionStatus

        mock_result = MagicMock()
        mock_result.checkpoint_name = "test<checkpoint>&'special\""
        mock_result.run_id = "run123"
        mock_result.run_time = datetime.now()
        mock_result.data_asset = "test<file>.csv"
        mock_result.validation_result = None
        mock_result.status = MagicMock(value="failure")

        action = OpsGenieAction(api_key="test-key", client=mock_http_client)
        result = action._execute(mock_result)

        assert result.status == ActionStatus.SUCCESS
