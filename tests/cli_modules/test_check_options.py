"""Tests for check command --exclude-columns and --validator-config options."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import typer
from typer.testing import CliRunner

from truthound.cli_modules.core.check import check_cmd


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def app():
    """Create Typer app with check command."""
    _app = typer.Typer()
    _app.command(name="check")(check_cmd)
    return _app


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "users.csv"
    csv_file.write_text(
        "id,first_name,last_name,age\n"
        "1,Alice,Smith,25\n"
        "2,Bob,Jones,30\n"
        "3,Alice,Brown,35\n"
        "4,David,Smith,40\n"
    )
    return csv_file


class TestExcludeColumns:
    """Tests for --exclude-columns option."""

    def test_exclude_columns_passed_to_api(self, runner, app, sample_csv):
        """Test that --exclude-columns is passed to check() API."""
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                str(sample_csv),
                "--exclude-columns", "first_name,last_name",
            ])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["exclude_columns"] == ["first_name", "last_name"]

    def test_exclude_columns_single(self, runner, app, sample_csv):
        """Test --exclude-columns with a single column."""
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                str(sample_csv),
                "--exclude-columns", "first_name",
            ])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["exclude_columns"] == ["first_name"]

    def test_no_exclude_columns(self, runner, app, sample_csv):
        """Test that exclude_columns is None when not provided."""
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [str(sample_csv)])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["exclude_columns"] is None


class TestValidatorConfig:
    """Tests for --validator-config option."""

    def test_validator_config_json_string(self, runner, app, sample_csv):
        """Test --validator-config with inline JSON string."""
        config = '{"unique": {"exclude_columns": ["first_name"]}}'
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                str(sample_csv),
                "--validator-config", config,
            ])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["validator_config"] == {
                "unique": {"exclude_columns": ["first_name"]}
            }

    def test_validator_config_json_file(self, runner, app, sample_csv, tmp_path):
        """Test --validator-config with a JSON file path."""
        config_data = {"unique": {"exclude_columns": ["first_name", "last_name"]}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                str(sample_csv),
                "--validator-config", str(config_file),
            ])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["validator_config"] == config_data

    def test_validator_config_invalid_json(self, runner, app, sample_csv):
        """Test --validator-config with invalid JSON string."""
        result = runner.invoke(app, [
            str(sample_csv),
            "--validator-config", "{invalid json}",
        ])

        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_validator_config_not_object(self, runner, app, sample_csv):
        """Test --validator-config with non-object JSON."""
        result = runner.invoke(app, [
            str(sample_csv),
            "--validator-config", '["not", "an", "object"]',
        ])

        assert result.exit_code == 1
        assert "must be a JSON object" in result.output

    def test_validator_config_file_not_found(self, runner, app, sample_csv):
        """Test --validator-config with non-existent file path."""
        result = runner.invoke(app, [
            str(sample_csv),
            "--validator-config", "nonexistent.json",
        ])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_validator_config(self, runner, app, sample_csv):
        """Test that validator_config is None when not provided."""
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [str(sample_csv)])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["validator_config"] is None


class TestCombinedOptions:
    """Tests for combined --exclude-columns and --validator-config."""

    def test_both_options_together(self, runner, app, sample_csv):
        """Test using both --exclude-columns and --validator-config."""
        config = '{"null": {"sample_size": 10}}'
        with patch("truthound.api.check") as mock_check:
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                str(sample_csv),
                "--exclude-columns", "first_name",
                "--validator-config", config,
                "--validators", "null,unique",
            ])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["exclude_columns"] == ["first_name"]
            assert call_kwargs["validator_config"] == {"null": {"sample_size": 10}}
            assert call_kwargs["validators"] == ["null", "unique"]
