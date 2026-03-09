"""Tests for the ``truthound read`` CLI command."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import typer
from typer.testing import CliRunner

from truthound.cli_modules.core.read import read_cmd


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def app():
    _app = typer.Typer()
    _app.command(name="read")(read_cmd)
    return _app


@pytest.fixture
def sample_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "id,name,age,city\n"
        "1,Alice,25,NYC\n"
        "2,Bob,30,LA\n"
        "3,Charlie,35,Chicago\n"
        "4,Diana,40,Boston\n"
        "5,Eve,28,Seattle\n"
    )
    return csv


@pytest.fixture
def sample_json(tmp_path):
    jf = tmp_path / "data.json"
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    jf.write_text(json.dumps(data))
    return jf


# =============================================================================
# Basic Read
# =============================================================================


class TestReadBasic:
    """Basic file reading tests."""

    def test_read_csv(self, runner, app, sample_csv):
        """Read CSV file outputs data."""
        result = runner.invoke(app, [str(sample_csv)])
        assert result.exit_code == 0
        assert "5 rows" in result.output or "Shape" in result.output

    def test_read_no_input_error(self, runner, app):
        """No input produces an error."""
        result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_read_nonexistent_file_error(self, runner, app, tmp_path):
        """Non-existent file produces an error."""
        fake = tmp_path / "missing.csv"
        result = runner.invoke(app, [str(fake)])
        assert result.exit_code != 0


# =============================================================================
# Row/Column Selection
# =============================================================================


class TestReadSelection:
    """Row and column selection tests."""

    def test_head(self, runner, app, sample_csv):
        """--head limits rows."""
        result = runner.invoke(app, [str(sample_csv), "--head", "2"])
        assert result.exit_code == 0
        assert "2 rows" in result.output or "Shape: 2" in result.output

    def test_columns(self, runner, app, sample_csv):
        """--columns selects specific columns."""
        result = runner.invoke(app, [str(sample_csv), "--columns", "id,name"])
        assert result.exit_code == 0
        assert "2 columns" in result.output or "x 2" in result.output

    def test_columns_missing_warns(self, runner, app, sample_csv):
        """Missing columns produce a warning."""
        result = runner.invoke(app, [str(sample_csv), "--columns", "id,nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_head_and_columns(self, runner, app, sample_csv):
        """--head and --columns together."""
        result = runner.invoke(app, [str(sample_csv), "--head", "3", "--columns", "name,age"])
        assert result.exit_code == 0

    def test_sample(self, runner, app, sample_csv):
        """--sample returns subset."""
        result = runner.invoke(app, [str(sample_csv), "--sample", "2"])
        assert result.exit_code == 0
        assert "2 rows" in result.output or "Shape: 2" in result.output


# =============================================================================
# Inspection Modes
# =============================================================================


class TestReadInspection:
    """Schema-only and count-only mode tests."""

    def test_schema_only(self, runner, app, sample_csv):
        """--schema-only shows column names and types."""
        result = runner.invoke(app, [str(sample_csv), "--schema-only"])
        assert result.exit_code == 0
        assert "Column" in result.output
        assert "Type" in result.output
        assert "id" in result.output
        assert "name" in result.output

    def test_count_only(self, runner, app, sample_csv):
        """--count-only shows just the row count."""
        result = runner.invoke(app, [str(sample_csv), "--count-only"])
        assert result.exit_code == 0
        assert "Rows:" in result.output
        assert "5" in result.output


# =============================================================================
# Output Formats
# =============================================================================


class TestReadFormats:
    """Output format tests."""

    def test_format_csv(self, runner, app, sample_csv):
        """--format csv outputs CSV text."""
        result = runner.invoke(app, [str(sample_csv), "--format", "csv", "--head", "2"])
        assert result.exit_code == 0
        assert "id,name,age,city" in result.output

    def test_format_json(self, runner, app, sample_csv):
        """--format json outputs valid JSON."""
        result = runner.invoke(app, [str(sample_csv), "--format", "json", "--head", "2"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Polars write_json output is valid JSON (format may vary by version)
        assert isinstance(data, (dict, list))

    def test_format_ndjson(self, runner, app, sample_csv):
        """--format ndjson outputs newline-delimited JSON."""
        result = runner.invoke(app, [str(sample_csv), "--format", "ndjson", "--head", "2"])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().split("\n") if l.strip()]
        assert len(lines) == 2

    def test_parquet_requires_output(self, runner, app, sample_csv):
        """--format parquet without --output is an error."""
        result = runner.invoke(app, [str(sample_csv), "--format", "parquet"])
        assert result.exit_code == 1
        assert "required" in result.output.lower()


# =============================================================================
# Output File
# =============================================================================


class TestReadOutput:
    """Output file tests."""

    def test_output_csv(self, runner, app, sample_csv, tmp_path):
        """--output writes CSV file."""
        out = tmp_path / "out.csv"
        result = runner.invoke(app, [str(sample_csv), "--output", str(out), "--head", "3"])
        assert result.exit_code == 0
        assert out.exists()
        assert "written to" in result.output
        content = out.read_text()
        assert "id" in content

    def test_output_json(self, runner, app, sample_csv, tmp_path):
        """--output with json format writes JSON file."""
        out = tmp_path / "out.json"
        result = runner.invoke(app, [
            str(sample_csv), "--output", str(out), "--format", "json", "--head", "2",
        ])
        assert result.exit_code == 0
        assert out.exists()

    def test_output_parquet(self, runner, app, sample_csv, tmp_path):
        """--output with parquet format writes Parquet file."""
        out = tmp_path / "out.parquet"
        result = runner.invoke(app, [
            str(sample_csv), "--output", str(out), "--format", "parquet",
        ])
        assert result.exit_code == 0
        assert out.exists()
        assert out.stat().st_size > 0


# =============================================================================
# DataSource Integration (mocked)
# =============================================================================


class TestReadWithConnection:
    """Test read command with mocked database connection."""

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_read_with_connection(self, mock_get_sql, runner, app):
        """--connection + --table uses DataSource."""
        import polars as pl

        mock_source = MagicMock()
        mock_source.name = "test_table"
        mock_lf = pl.LazyFrame({"id": [1, 2], "name": ["a", "b"]})
        mock_source.to_polars_lazyframe.return_value = mock_lf
        mock_get_sql.return_value = mock_source

        result = runner.invoke(app, [
            "--connection", "postgresql://user:pass@host/db",
            "--table", "users",
        ])
        assert result.exit_code == 0
        assert "2 rows" in result.output or "Shape" in result.output

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_read_schema_only_with_connection(self, mock_get_sql, runner, app):
        """--schema-only works with database source."""
        import polars as pl

        mock_source = MagicMock()
        mock_source.name = "test_table"
        mock_lf = pl.LazyFrame({"id": [1], "name": ["a"]})
        mock_source.to_polars_lazyframe.return_value = mock_lf
        mock_get_sql.return_value = mock_source

        result = runner.invoke(app, [
            "--connection", "postgresql://user:pass@host/db",
            "--table", "users",
            "--schema-only",
        ])
        assert result.exit_code == 0
        assert "id" in result.output
        assert "name" in result.output
