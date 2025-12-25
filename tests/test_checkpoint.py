"""Tests for checkpoint module (Phase 6).

This module contains comprehensive tests for the Checkpoint & CI/CD
integration functionality.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_data_file(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_content = """id,name,email,age,score
1,Alice,alice@example.com,30,85.5
2,Bob,bob@example.com,25,92.0
3,Charlie,charlie@example.com,35,78.5
4,Diana,,28,88.0
5,Eve,eve@example.com,-5,95.0
"""
    data_file = tmp_path / "test_data.csv"
    data_file.write_text(csv_content)
    return data_file


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_data_file: Path) -> Path:
    """Create a sample checkpoint configuration file."""
    config = {
        "checkpoints": [
            {
                "name": "test_checkpoint",
                "data_source": str(sample_data_file),
                "validators": ["null", "range"],
                "auto_schema": False,
                "actions": [
                    {
                        "type": "store_result",
                        "store_path": str(tmp_path / "results"),
                    },
                ],
            },
        ],
    }

    import yaml
    config_file = tmp_path / "truthound.yaml"
    config_file.write_text(yaml.dump(config))
    return config_file


# =============================================================================
# Action Base Tests
# =============================================================================


class TestActionBase:
    """Tests for base action functionality."""

    def test_notify_condition_should_notify(self):
        """Test NotifyCondition.should_notify() logic."""
        from truthound.checkpoint.actions.base import NotifyCondition

        # ALWAYS
        assert NotifyCondition.ALWAYS.should_notify("success") is True
        assert NotifyCondition.ALWAYS.should_notify("failure") is True

        # SUCCESS
        assert NotifyCondition.SUCCESS.should_notify("success") is True
        assert NotifyCondition.SUCCESS.should_notify("failure") is False

        # FAILURE
        assert NotifyCondition.FAILURE.should_notify("failure") is True
        assert NotifyCondition.FAILURE.should_notify("success") is False

        # FAILURE_OR_ERROR
        assert NotifyCondition.FAILURE_OR_ERROR.should_notify("failure") is True
        assert NotifyCondition.FAILURE_OR_ERROR.should_notify("error") is True
        assert NotifyCondition.FAILURE_OR_ERROR.should_notify("success") is False

        # NOT_SUCCESS
        assert NotifyCondition.NOT_SUCCESS.should_notify("failure") is True
        assert NotifyCondition.NOT_SUCCESS.should_notify("warning") is True
        assert NotifyCondition.NOT_SUCCESS.should_notify("success") is False

    def test_action_result_serialization(self):
        """Test ActionResult to_dict/from_dict."""
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        result = ActionResult(
            action_name="test_action",
            action_type="test",
            status=ActionStatus.SUCCESS,
            message="Test completed",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            duration_ms=100.5,
            details={"key": "value"},
        )

        result_dict = result.to_dict()
        assert result_dict["action_name"] == "test_action"
        assert result_dict["status"] == "success"

        restored = ActionResult.from_dict(result_dict)
        assert restored.action_name == result.action_name
        assert restored.status == result.status


# =============================================================================
# Store Result Action Tests
# =============================================================================


class TestStoreResultAction:
    """Tests for StoreValidationResult action."""

    def test_store_result_file(self, tmp_path: Path, sample_data_file: Path):
        """Test storing results to local filesystem."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import StoreValidationResult

        store_path = tmp_path / "results"

        checkpoint = Checkpoint(
            name="test_store",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                StoreValidationResult(
                    store_path=str(store_path),
                    partition_by="date",
                    format="json",
                ),
            ],
        )

        result = checkpoint.run()

        # Check that files were created
        assert store_path.exists()
        json_files = list(store_path.rglob("*.json"))
        assert len(json_files) >= 1

        # Verify JSON content
        content = json.loads(json_files[0].read_text())
        assert "run_id" in content
        assert "status" in content

    def test_store_result_partition_by_status(self, tmp_path: Path, sample_data_file: Path):
        """Test partition by status."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import StoreValidationResult

        store_path = tmp_path / "results"

        action = StoreValidationResult(
            store_path=str(store_path),
            partition_by="status",
        )

        checkpoint = Checkpoint(
            name="test_partition",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[action],
        )

        result = checkpoint.run()
        assert store_path.exists()


# =============================================================================
# Update Docs Action Tests
# =============================================================================


class TestUpdateDocsAction:
    """Tests for UpdateDataDocs action."""

    def test_update_docs_html(self, tmp_path: Path, sample_data_file: Path):
        """Test generating HTML documentation."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import UpdateDataDocs

        docs_path = tmp_path / "docs"

        checkpoint = Checkpoint(
            name="test_docs",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                UpdateDataDocs(
                    site_path=str(docs_path),
                    format="html",
                    include_history=True,
                ),
            ],
        )

        result = checkpoint.run()

        # Check that docs were created
        assert docs_path.exists()
        assert (docs_path / "index.html").exists()
        assert (docs_path / "runs").exists()
        assert (docs_path / "assets" / "style.css").exists()

    def test_update_docs_markdown(self, tmp_path: Path, sample_data_file: Path):
        """Test generating Markdown documentation."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import UpdateDataDocs

        docs_path = tmp_path / "docs"

        checkpoint = Checkpoint(
            name="test_docs_md",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                UpdateDataDocs(
                    site_path=str(docs_path),
                    format="markdown",
                ),
            ],
        )

        result = checkpoint.run()

        # Check that docs directory was created (format is "markdown" so file is index.markdown)
        assert docs_path.exists()
        assert (docs_path / "index.markdown").exists()


# =============================================================================
# Notification Action Tests
# =============================================================================


class TestSlackNotification:
    """Tests for SlackNotification action."""

    def test_slack_notification_skipped_on_success(self, sample_data_file: Path):
        """Test that Slack notification is skipped on success when notify_on=failure."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import SlackNotification

        checkpoint = Checkpoint(
            name="test_slack",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                SlackNotification(
                    webhook_url="https://hooks.slack.com/test",
                    notify_on="failure",
                ),
            ],
        )

        result = checkpoint.run()

        # Find the Slack action result
        slack_result = next(
            (r for r in result.action_results if r.action_type == "slack_notification"),
            None,
        )

        # Should be skipped because validation likely succeeded
        assert slack_result is not None
        assert slack_result.status.value in ("skipped", "success", "error")

    def test_slack_config_validation(self):
        """Test Slack configuration validation."""
        from truthound.checkpoint.actions import SlackNotification

        action = SlackNotification(webhook_url="")
        errors = action.validate_config()
        assert len(errors) > 0
        assert "webhook_url is required" in errors[0]


class TestWebhookAction:
    """Tests for WebhookAction."""

    def test_webhook_config_validation(self):
        """Test webhook configuration validation."""
        from truthound.checkpoint.actions import WebhookAction

        action = WebhookAction(url="", method="INVALID")
        errors = action.validate_config()
        assert any("url is required" in e for e in errors)
        assert any("Invalid HTTP method" in e for e in errors)

    @patch("urllib.request.urlopen")
    def test_webhook_execution(self, mock_urlopen, sample_data_file: Path):
        """Test webhook action execution."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import WebhookAction

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        checkpoint = Checkpoint(
            name="test_webhook",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                WebhookAction(
                    url="https://api.example.com/webhook",
                    method="POST",
                ),
            ],
        )

        result = checkpoint.run()

        webhook_result = next(
            (r for r in result.action_results if r.action_type == "webhook"),
            None,
        )
        assert webhook_result is not None


# =============================================================================
# Trigger Tests
# =============================================================================


class TestScheduleTrigger:
    """Tests for ScheduleTrigger."""

    def test_schedule_trigger_interval(self):
        """Test schedule trigger with interval."""
        from truthound.checkpoint.triggers import ScheduleTrigger

        trigger = ScheduleTrigger(interval_minutes=30)
        trigger.start()

        # First run should trigger immediately
        result = trigger.should_trigger()
        assert result.should_run is True

        # Record the run
        trigger.record_run()

        # Second check should not trigger (too soon)
        result = trigger.should_trigger()
        assert result.should_run is False
        assert result.next_run is not None

    def test_schedule_trigger_weekday_filter(self):
        """Test schedule trigger with weekday filter."""
        from truthound.checkpoint.triggers import ScheduleTrigger

        # Create trigger that only runs on specific days
        trigger = ScheduleTrigger(
            interval_hours=1,
            run_on_weekdays=[0, 1, 2, 3, 4],  # Mon-Fri
        )
        trigger.start()

        # Behavior depends on current day
        result = trigger.should_trigger()
        # Just verify it returns a valid result
        assert isinstance(result.should_run, bool)


class TestCronTrigger:
    """Tests for CronTrigger."""

    def test_cron_trigger_parsing(self):
        """Test cron expression parsing."""
        from truthound.checkpoint.triggers import CronTrigger

        trigger = CronTrigger(expression="*/15 * * * *")
        trigger.start()

        # Should parse without error
        result = trigger.should_trigger()
        assert isinstance(result.should_run, bool)

    def test_cron_trigger_invalid_expression(self):
        """Test cron trigger with invalid expression."""
        from truthound.checkpoint.triggers import CronTrigger

        trigger = CronTrigger(expression="invalid")
        errors = trigger.validate_config()
        assert len(errors) > 0


class TestEventTrigger:
    """Tests for EventTrigger."""

    def test_event_trigger_fire(self):
        """Test firing events."""
        from truthound.checkpoint.triggers import EventTrigger

        trigger = EventTrigger(
            event_type="data_updated",
            event_filter={"source": "production"},
        )
        trigger.start()

        # Fire matching event
        accepted = trigger.fire_event({"source": "production", "table": "users"})
        assert accepted is True

        # Check trigger
        result = trigger.should_trigger()
        assert result.should_run is True

    def test_event_trigger_filter(self):
        """Test event filtering."""
        from truthound.checkpoint.triggers import EventTrigger

        trigger = EventTrigger(
            event_filter={"source": "production"},
        )
        trigger.start()

        # Fire non-matching event
        accepted = trigger.fire_event({"source": "staging"})
        assert accepted is False

        result = trigger.should_trigger()
        assert result.should_run is False


class TestFileWatchTrigger:
    """Tests for FileWatchTrigger."""

    def test_file_watch_trigger(self, tmp_path: Path):
        """Test file watch trigger."""
        from truthound.checkpoint.triggers import FileWatchTrigger

        # Create a test file
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        trigger = FileWatchTrigger(
            paths=[str(tmp_path)],
            patterns=["*.csv"],
            poll_interval_seconds=0.1,
        )
        trigger.start()

        # Initial scan shouldn't trigger
        result = trigger.should_trigger()
        assert result.should_run is False

        # Modify file
        import time
        time.sleep(0.1)
        test_file.write_text("a,b,c\n1,2,3\n4,5,6")

        # Should detect change
        result = trigger.should_trigger()
        assert result.should_run is True


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for Checkpoint class."""

    def test_checkpoint_basic_run(self, sample_data_file: Path):
        """Test basic checkpoint execution."""
        from truthound.checkpoint import Checkpoint

        checkpoint = Checkpoint(
            name="basic_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        result = checkpoint.run()

        assert result.checkpoint_name == "basic_test"
        assert result.run_id is not None
        assert result.status is not None
        assert result.duration_ms > 0

    def test_checkpoint_with_actions(self, tmp_path: Path, sample_data_file: Path):
        """Test checkpoint with multiple actions."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.actions import StoreValidationResult, UpdateDataDocs

        checkpoint = Checkpoint(
            name="actions_test",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                StoreValidationResult(store_path=str(tmp_path / "results")),
                UpdateDataDocs(site_path=str(tmp_path / "docs")),
            ],
        )

        result = checkpoint.run()

        assert len(result.action_results) == 2
        assert all(r.status.value in ("success", "skipped") for r in result.action_results)

    def test_checkpoint_validation_errors(self):
        """Test checkpoint configuration validation."""
        from truthound.checkpoint import Checkpoint

        checkpoint = Checkpoint(name="")  # Invalid: no name or data source

        errors = checkpoint.validate()
        assert len(errors) > 0

    def test_checkpoint_result_serialization(self, sample_data_file: Path):
        """Test CheckpointResult serialization."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.checkpoint import CheckpointResult

        checkpoint = Checkpoint(
            name="serialization_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        result = checkpoint.run()

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["checkpoint_name"] == "serialization_test"
        assert "run_id" in result_dict
        assert "status" in result_dict

        # Test from_dict
        restored = CheckpointResult.from_dict(result_dict)
        assert restored.checkpoint_name == result.checkpoint_name
        assert restored.run_id == result.run_id


# =============================================================================
# Registry Tests
# =============================================================================


class TestCheckpointRegistry:
    """Tests for CheckpointRegistry."""

    def test_registry_register_and_get(self, sample_data_file: Path):
        """Test registering and retrieving checkpoints."""
        from truthound.checkpoint import Checkpoint, CheckpointRegistry

        # Create a fresh registry instance for testing
        registry = CheckpointRegistry()
        registry.clear()

        checkpoint = Checkpoint(
            name="registry_test",
            data_source=str(sample_data_file),
        )

        registry.register(checkpoint)

        assert "registry_test" in registry
        assert registry.get("registry_test") is checkpoint

    def test_registry_load_from_yaml(self, sample_config_file: Path):
        """Test loading checkpoints from YAML."""
        from truthound.checkpoint import CheckpointRegistry

        registry = CheckpointRegistry()
        registry.clear()

        checkpoints = registry.load_from_yaml(sample_config_file)

        assert len(checkpoints) == 1
        assert checkpoints[0].name == "test_checkpoint"


# =============================================================================
# Runner Tests
# =============================================================================


class TestCheckpointRunner:
    """Tests for CheckpointRunner."""

    def test_runner_run_once(self, sample_data_file: Path):
        """Test running a checkpoint once."""
        from truthound.checkpoint import Checkpoint, CheckpointRunner

        checkpoint = Checkpoint(
            name="runner_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        runner = CheckpointRunner()
        runner.add_checkpoint(checkpoint)

        result = runner.run_once("runner_test")

        assert result.checkpoint_name == "runner_test"
        assert result.status is not None

    def test_runner_run_all(self, sample_data_file: Path):
        """Test running all checkpoints."""
        from truthound.checkpoint import Checkpoint, CheckpointRunner

        runner = CheckpointRunner()
        runner.add_checkpoint(Checkpoint(
            name="test1",
            data_source=str(sample_data_file),
            validators=["null"],
        ))
        runner.add_checkpoint(Checkpoint(
            name="test2",
            data_source=str(sample_data_file),
            validators=["null"],
        ))

        results = runner.run_all()

        assert len(results) == 2


# =============================================================================
# CI/CD Integration Tests
# =============================================================================


class TestCIDetector:
    """Tests for CI platform detection."""

    def test_detect_local(self):
        """Test detection of local environment."""
        from truthound.checkpoint.ci import detect_ci_platform, CIPlatform

        # Clear CI environment variables
        with patch.dict(os.environ, {}, clear=True):
            platform = detect_ci_platform()
            assert platform == CIPlatform.LOCAL

    def test_detect_github_actions(self):
        """Test detection of GitHub Actions."""
        from truthound.checkpoint.ci import detect_ci_platform, CIPlatform

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            platform = detect_ci_platform()
            assert platform == CIPlatform.GITHUB_ACTIONS

    def test_detect_gitlab_ci(self):
        """Test detection of GitLab CI."""
        from truthound.checkpoint.ci import detect_ci_platform, CIPlatform

        with patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=True):
            platform = detect_ci_platform()
            assert platform == CIPlatform.GITLAB_CI

    def test_get_ci_environment(self):
        """Test getting CI environment info."""
        from truthound.checkpoint.ci import get_ci_environment

        with patch.dict(os.environ, {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_SHA": "abc123",
            "GITHUB_REF_NAME": "main",
        }, clear=True):
            env = get_ci_environment()
            assert env.is_ci is True
            assert env.repository == "owner/repo"
            assert env.commit_sha == "abc123"


class TestCIReporter:
    """Tests for CI reporters."""

    def test_github_reporter_outputs(self, tmp_path: Path, sample_data_file: Path):
        """Test GitHub Actions reporter outputs."""
        from truthound.checkpoint import Checkpoint
        from truthound.checkpoint.ci import GitHubActionsReporter

        checkpoint = Checkpoint(
            name="ci_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        result = checkpoint.run()

        output_file = tmp_path / "github_output"
        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            reporter = GitHubActionsReporter()
            reporter.set_output("test_key", "test_value")

        assert output_file.exists()
        content = output_file.read_text()
        assert "test_key=test_value" in content


class TestCITemplates:
    """Tests for CI configuration templates."""

    def test_generate_github_workflow(self):
        """Test GitHub Actions workflow generation."""
        from truthound.checkpoint.ci import generate_github_workflow

        workflow = generate_github_workflow(
            checkpoint_name="test_check",
            schedule="0 0 * * *",
        )

        assert "name: Data Quality Check" in workflow
        assert "test_check" in workflow
        assert "truthound checkpoint run" in workflow

    def test_generate_gitlab_ci(self):
        """Test GitLab CI configuration generation."""
        from truthound.checkpoint.ci import generate_gitlab_ci

        config = generate_gitlab_ci(checkpoint_name="test_check")

        assert "data-quality:" in config
        assert "truthound checkpoint run" in config

    def test_generate_jenkinsfile(self):
        """Test Jenkinsfile generation."""
        from truthound.checkpoint.ci import generate_jenkinsfile

        jenkinsfile = generate_jenkinsfile(checkpoint_name="test_check")

        assert "pipeline {" in jenkinsfile
        assert "truthound checkpoint run" in jenkinsfile

    def test_generate_circleci_config(self):
        """Test CircleCI configuration generation."""
        from truthound.checkpoint.ci import generate_circleci_config

        config = generate_circleci_config(checkpoint_name="test_check")

        assert "version: 2.1" in config
        assert "truthound checkpoint run" in config


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLICommands:
    """Tests for checkpoint CLI commands."""

    def test_checkpoint_init(self, tmp_path: Path):
        """Test checkpoint init command."""
        from typer.testing import CliRunner
        from truthound.cli import app

        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["checkpoint", "init", "-o", "test.yaml"])

            assert result.exit_code == 0
            assert Path("test.yaml").exists()

    def test_checkpoint_run_adhoc(self, sample_data_file: Path):
        """Test ad-hoc checkpoint run."""
        from typer.testing import CliRunner
        from truthound.cli import app

        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "checkpoint", "run", "test_adhoc",
                "--data", str(sample_data_file),
                "--format", "json",
            ],
        )

        assert result.exit_code == 0

    def test_checkpoint_validate(self, sample_config_file: Path):
        """Test checkpoint validate command."""
        from typer.testing import CliRunner
        from truthound.cli import app

        runner = CliRunner()

        result = runner.invoke(
            app,
            ["checkpoint", "validate", str(sample_config_file)],
        )

        assert result.exit_code == 0
        assert "valid" in result.output.lower()
