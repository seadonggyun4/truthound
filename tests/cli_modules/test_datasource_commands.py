"""Tests for DataSource support across CLI commands.

Verifies that scan, mask, profile, learn, and compare commands
correctly accept and pass through database connection options.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import typer
from typer.testing import CliRunner

from truthound.cli_modules.core.check import check_cmd
from truthound.cli_modules.core.scan import scan_cmd
from truthound.cli_modules.core.mask import mask_cmd
from truthound.cli_modules.core.profile import profile_cmd
from truthound.cli_modules.core.learn import learn_cmd
from truthound.cli_modules.core.compare import compare_cmd


@pytest.fixture
def runner():
    return CliRunner()


def _make_app(cmd, name):
    app = typer.Typer()
    app.command(name=name)(cmd)
    return app


@pytest.fixture
def sample_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("id,name,age\n1,Alice,25\n2,Bob,30\n")
    return csv


def _mock_sql_source(table_name="users"):
    """Create a mock SQL data source returning a small DataFrame."""
    source = MagicMock()
    source.name = table_name
    lf = pl.LazyFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    source.to_polars_lazyframe.return_value = lf
    return source


# =============================================================================
# Check with DataSource
# =============================================================================


class TestCheckWithDatasource:
    """Test check command accepts datasource options."""

    def test_check_with_connection(self, runner, sample_csv):
        """--connection passes source= to check API."""
        app = _make_app(check_cmd, "check")

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.api.check") as mock_check,
        ):
            mock_sql.return_value = _mock_sql_source()
            mock_report = MagicMock()
            mock_report.has_issues = False
            mock_report.exception_summary = None
            mock_check.return_value = mock_report

            result = runner.invoke(app, [
                "--connection", "postgresql://user:pass@host/db",
                "--table", "users",
            ])

            assert result.exit_code == 0
            # Verify source= was passed (not data path)
            assert mock_check.call_args[1].get("source") is not None or \
                   mock_check.call_args.kwargs.get("source") is not None

    def test_check_file_and_connection_mutually_exclusive(self, runner, sample_csv):
        """file + --connection raises error."""
        app = _make_app(check_cmd, "check")

        result = runner.invoke(app, [
            str(sample_csv),
            "--connection", "postgresql://host/db",
            "--table", "t",
        ])
        assert result.exit_code != 0


# =============================================================================
# Scan with DataSource
# =============================================================================


class TestScanWithDatasource:
    """Test scan command accepts datasource options."""

    def test_scan_with_connection(self, runner):
        """--connection passes source= to scan API."""
        app = _make_app(scan_cmd, "scan")

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.api.scan") as mock_scan,
        ):
            mock_sql.return_value = _mock_sql_source()
            mock_report = MagicMock()
            mock_scan.return_value = mock_report
            mock_report.print = MagicMock()

            result = runner.invoke(app, [
                "--connection", "postgresql://user:pass@host/db",
                "--table", "users",
            ])

            assert result.exit_code == 0
            mock_scan.assert_called_once()
            call_kwargs = mock_scan.call_args
            # scan(source=source)
            assert call_kwargs.kwargs.get("source") is not None

    def test_scan_no_input_error(self, runner):
        """No input produces an error."""
        app = _make_app(scan_cmd, "scan")
        result = runner.invoke(app, [])
        assert result.exit_code != 0


# =============================================================================
# Mask with DataSource
# =============================================================================


class TestMaskWithDatasource:
    """Test mask command accepts datasource options."""

    def test_mask_with_connection(self, runner, tmp_path):
        """--connection passes source= to mask API."""
        app = _make_app(mask_cmd, "mask")
        out = tmp_path / "masked.csv"

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.api.mask") as mock_mask,
        ):
            mock_sql.return_value = _mock_sql_source()
            mock_df = pl.DataFrame({"id": [1, 2], "name": ["***", "***"]})
            mock_mask.return_value = mock_df

            result = runner.invoke(app, [
                "--connection", "postgresql://user:pass@host/db",
                "--table", "users",
                "--output", str(out),
            ])

            assert result.exit_code == 0
            assert out.exists()
            mock_mask.assert_called_once()


# =============================================================================
# Profile with DataSource
# =============================================================================


class TestProfileWithDatasource:
    """Test profile command accepts datasource options."""

    def test_profile_with_connection(self, runner):
        """--connection passes source= to profile API."""
        app = _make_app(profile_cmd, "profile")

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.api.profile") as mock_profile,
        ):
            mock_sql.return_value = _mock_sql_source()
            mock_report = MagicMock()
            mock_profile.return_value = mock_report
            mock_report.print = MagicMock()

            result = runner.invoke(app, [
                "--connection", "postgresql://user:pass@host/db",
                "--table", "users",
            ])

            assert result.exit_code == 0
            mock_profile.assert_called_once()
            assert mock_profile.call_args.kwargs.get("source") is not None


# =============================================================================
# Learn with DataSource
# =============================================================================


class TestLearnWithDatasource:
    """Test learn command accepts datasource options."""

    def test_learn_with_connection(self, runner, tmp_path):
        """--connection passes source= to learn API."""
        app = _make_app(learn_cmd, "learn")
        out = tmp_path / "schema.yaml"

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.schema.learn") as mock_learn,
        ):
            mock_sql.return_value = _mock_sql_source()
            mock_schema = MagicMock()
            mock_schema.columns = ["id", "name", "age"]
            mock_schema.row_count = 2
            mock_learn.return_value = mock_schema

            result = runner.invoke(app, [
                "--connection", "postgresql://user:pass@host/db",
                "--table", "users",
                "--output", str(out),
            ])

            assert result.exit_code == 0
            mock_learn.assert_called_once()
            assert mock_learn.call_args.kwargs.get("source") is not None


# =============================================================================
# Compare with DataSource Config
# =============================================================================


class TestCompareWithDatasource:
    """Test compare command accepts --source-config for dual sources."""

    def test_compare_with_source_config(self, runner, tmp_path):
        """--source-config with baseline/current sections works."""
        app = _make_app(compare_cmd, "compare")

        cfg = tmp_path / "drift.yaml"
        cfg.write_text(
            "baseline:\n"
            "  connection: 'postgresql://host/db'\n"
            "  table: train\n"
            "current:\n"
            "  connection: 'postgresql://host/db'\n"
            "  table: prod\n"
        )

        with (
            patch("truthound.datasources.factory.get_sql_datasource") as mock_sql,
            patch("truthound.drift.compare") as mock_compare,
        ):
            source_b = MagicMock()
            source_c = MagicMock()
            lf_b = pl.LazyFrame({"x": [1, 2, 3]})
            lf_c = pl.LazyFrame({"x": [4, 5, 6]})
            source_b.to_polars_lazyframe.return_value = lf_b
            source_c.to_polars_lazyframe.return_value = lf_c
            mock_sql.side_effect = [source_b, source_c]

            mock_report = MagicMock()
            mock_report.has_drift = False
            mock_compare.return_value = mock_report
            mock_report.print = MagicMock()

            result = runner.invoke(app, [
                "--source-config", str(cfg),
            ])

            assert result.exit_code == 0
            mock_compare.assert_called_once()

    def test_compare_files_still_works(self, runner, tmp_path):
        """Positional file arguments still work."""
        app = _make_app(compare_cmd, "compare")

        f1 = tmp_path / "base.csv"
        f2 = tmp_path / "curr.csv"
        f1.write_text("x\n1\n2\n3\n")
        f2.write_text("x\n4\n5\n6\n")

        with patch("truthound.drift.compare") as mock_compare:
            mock_report = MagicMock()
            mock_report.has_drift = False
            mock_compare.return_value = mock_report
            mock_report.print = MagicMock()

            result = runner.invoke(app, [str(f1), str(f2)])
            assert result.exit_code == 0
            mock_compare.assert_called_once()
