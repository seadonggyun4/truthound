"""Tests for the shared DataSource resolution layer.

Tests cover: resolve_datasource(), resolve_compare_sources(),
parse_source_config(), create_datasource_from_config(), and
input validation logic.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from truthound.cli_modules.common.datasource import (
    create_datasource_from_config,
    parse_source_config,
    resolve_compare_sources,
    resolve_datasource,
)
from truthound.cli_modules.common.errors import DataSourceError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    csv = tmp_path / "data.csv"
    csv.write_text("id,name\n1,Alice\n2,Bob\n")
    return csv


@pytest.fixture
def source_config_json(tmp_path):
    """Create a JSON source config file."""
    cfg = tmp_path / "source.json"
    cfg.write_text(json.dumps({
        "connection": "postgresql://user:pass@host:5432/db",
        "table": "users",
    }))
    return cfg


@pytest.fixture
def source_config_yaml(tmp_path):
    """Create a YAML source config file."""
    cfg = tmp_path / "source.yaml"
    cfg.write_text(
        "connection: 'postgresql://user:pass@host:5432/db'\n"
        "table: users\n"
    )
    return cfg


@pytest.fixture
def compare_config_yaml(tmp_path):
    """Create a YAML compare config file with baseline/current sections."""
    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        "baseline:\n"
        "  connection: 'postgresql://user:pass@host/db'\n"
        "  table: train_data\n"
        "current:\n"
        "  connection: 'postgresql://user:pass@host/db'\n"
        "  table: prod_data\n"
    )
    return cfg


# =============================================================================
# resolve_datasource
# =============================================================================


class TestResolveDatasource:
    """Tests for resolve_datasource()."""

    def test_file_only_returns_path(self, sample_csv):
        """File-only input returns (str_path, None)."""
        data_path, source = resolve_datasource(file=sample_csv)
        assert data_path == str(sample_csv)
        assert source is None

    def test_no_input_raises_error(self):
        """No input raises DataSourceError."""
        with pytest.raises(DataSourceError, match="No data input specified"):
            resolve_datasource()

    def test_file_not_found_raises_error(self, tmp_path):
        """Non-existent file raises error."""
        fake = tmp_path / "nonexistent.csv"
        with pytest.raises(Exception):
            resolve_datasource(file=fake)

    def test_file_and_connection_mutually_exclusive(self, sample_csv):
        """Providing both file and connection raises error."""
        with pytest.raises(DataSourceError, match="Conflicting"):
            resolve_datasource(file=sample_csv, connection="postgresql://host/db")

    def test_file_and_source_config_mutually_exclusive(self, sample_csv, source_config_json):
        """Providing both file and source_config raises error."""
        with pytest.raises(DataSourceError, match="Conflicting"):
            resolve_datasource(file=sample_csv, source_config=source_config_json)

    def test_connection_and_source_config_mutually_exclusive(self, source_config_json):
        """Providing both connection and source_config raises error."""
        with pytest.raises(DataSourceError, match="Conflicting"):
            resolve_datasource(
                connection="postgresql://host/db",
                source_config=source_config_json,
            )

    def test_connection_without_table_raises_error(self):
        """Connection without table or query raises error."""
        with pytest.raises(DataSourceError, match="--table or --query"):
            resolve_datasource(connection="postgresql://user:pass@host/db")

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_connection_with_table_returns_source(self, mock_get_sql):
        """Connection + table returns (None, source)."""
        mock_source = MagicMock()
        mock_get_sql.return_value = mock_source

        data_path, source = resolve_datasource(
            connection="postgresql://user:pass@host/db",
            table="users",
        )
        assert data_path is None
        assert source is mock_source
        mock_get_sql.assert_called_once_with(
            "postgresql://user:pass@host/db", table="users", query=None
        )

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_connection_with_query_returns_source(self, mock_get_sql):
        """Connection + query returns (None, source)."""
        mock_source = MagicMock()
        mock_get_sql.return_value = mock_source

        data_path, source = resolve_datasource(
            connection="postgresql://user:pass@host/db",
            query="SELECT * FROM orders WHERE date > '2024-01-01'",
        )
        assert data_path is None
        assert source is mock_source

    @patch("truthound.cli_modules.common.datasource.create_datasource_from_config")
    @patch("truthound.cli_modules.common.datasource.parse_source_config")
    def test_source_config_returns_source(self, mock_parse, mock_create, source_config_json):
        """Source config file returns (None, source)."""
        mock_config = {"connection": "postgresql://...", "table": "users"}
        mock_parse.return_value = mock_config
        mock_source = MagicMock()
        mock_create.return_value = mock_source

        data_path, source = resolve_datasource(source_config=source_config_json)
        assert data_path is None
        assert source is mock_source
        mock_parse.assert_called_once_with(source_config_json)
        mock_create.assert_called_once_with(mock_config)

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_source_name_applied(self, mock_get_sql):
        """--source-name is applied to the data source."""
        mock_source = MagicMock()
        mock_source.config = MagicMock()
        mock_get_sql.return_value = mock_source

        resolve_datasource(
            connection="postgresql://host/db",
            table="users",
            source_name="my-label",
        )
        # source_name should have been set
        assert mock_source.config.name == "my-label"


# =============================================================================
# resolve_compare_sources
# =============================================================================


class TestResolveCompareSources:
    """Tests for resolve_compare_sources()."""

    def test_two_files_returns_paths(self, tmp_path):
        """Two file paths return ((path1, None), (path2, None))."""
        f1 = tmp_path / "base.csv"
        f2 = tmp_path / "curr.csv"
        f1.write_text("a\n1\n")
        f2.write_text("a\n2\n")

        (bp, bs), (cp, cs) = resolve_compare_sources(baseline=f1, current=f2)
        assert bp == str(f1) and bs is None
        assert cp == str(f2) and cs is None

    def test_missing_one_file_raises_error(self, tmp_path):
        """Only one file provided raises error."""
        f1 = tmp_path / "base.csv"
        f1.write_text("a\n1\n")

        with pytest.raises(DataSourceError, match="Both baseline and current"):
            resolve_compare_sources(baseline=f1)

    def test_no_files_no_config_raises_error(self):
        """No arguments raises error."""
        with pytest.raises(DataSourceError, match="Both baseline and current"):
            resolve_compare_sources()

    def test_files_and_config_raises_error(self, tmp_path, compare_config_yaml):
        """Files + config raises error."""
        f1 = tmp_path / "base.csv"
        f1.write_text("a\n1\n")

        with pytest.raises(DataSourceError, match="Cannot specify both"):
            resolve_compare_sources(baseline=f1, source_config=compare_config_yaml)

    @patch("truthound.cli_modules.common.datasource.create_datasource_from_config")
    @patch("truthound.cli_modules.common.datasource.parse_source_config")
    def test_config_returns_dual_sources(self, mock_parse, mock_create, compare_config_yaml):
        """Config file with baseline/current returns two sources."""
        mock_parse.return_value = {
            "baseline": {"connection": "pg://...", "table": "train"},
            "current": {"connection": "pg://...", "table": "prod"},
        }
        mock_source_b = MagicMock()
        mock_source_c = MagicMock()
        mock_create.side_effect = [mock_source_b, mock_source_c]

        (bp, bs), (cp, cs) = resolve_compare_sources(source_config=compare_config_yaml)
        assert bp is None and bs is mock_source_b
        assert cp is None and cs is mock_source_c

    @patch("truthound.cli_modules.common.datasource.parse_source_config")
    def test_config_missing_baseline_raises_error(self, mock_parse, compare_config_yaml):
        """Config missing baseline section raises error."""
        mock_parse.return_value = {"current": {"connection": "pg://...", "table": "t"}}

        with pytest.raises(DataSourceError, match="baseline.*current"):
            resolve_compare_sources(source_config=compare_config_yaml)


# =============================================================================
# parse_source_config
# =============================================================================


class TestParseSourceConfig:
    """Tests for parse_source_config()."""

    def test_parse_json(self, tmp_path):
        """JSON config file is parsed correctly."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"connection": "pg://host/db", "table": "t"}))

        result = parse_source_config(cfg)
        assert result["connection"] == "pg://host/db"
        assert result["table"] == "t"

    def test_parse_yaml(self, tmp_path):
        """YAML config file is parsed correctly."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("connection: 'pg://host/db'\ntable: t\n")

        result = parse_source_config(cfg)
        assert result["connection"] == "pg://host/db"
        assert result["table"] == "t"

    def test_parse_yml(self, tmp_path):
        """YML extension also works."""
        cfg = tmp_path / "cfg.yml"
        cfg.write_text("connection: 'pg://host/db'\ntable: t\n")

        result = parse_source_config(cfg)
        assert result["table"] == "t"

    def test_invalid_json_raises_error(self, tmp_path):
        """Malformed JSON raises DataSourceError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text("{invalid json}")

        with pytest.raises(DataSourceError, match="Invalid JSON"):
            parse_source_config(cfg)

    def test_non_dict_raises_error(self, tmp_path):
        """Non-dict JSON content raises DataSourceError."""
        cfg = tmp_path / "arr.json"
        cfg.write_text('["a", "b"]')

        with pytest.raises(DataSourceError, match="must be a JSON/YAML object"):
            parse_source_config(cfg)

    def test_unsupported_extension_raises_error(self, tmp_path):
        """Unsupported file extension raises DataSourceError."""
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("[table]\nname = 'x'")

        with pytest.raises(DataSourceError, match="Unsupported config file format"):
            parse_source_config(cfg)


# =============================================================================
# create_datasource_from_config
# =============================================================================


class TestCreateDatasourceFromConfig:
    """Tests for create_datasource_from_config()."""

    @patch("truthound.datasources.factory.get_sql_datasource")
    def test_connection_string_style(self, mock_get_sql):
        """Config with 'connection' delegates to get_sql_datasource."""
        mock_source = MagicMock()
        mock_get_sql.return_value = mock_source

        result = create_datasource_from_config({
            "connection": "postgresql://host/db",
            "table": "users",
        })
        assert result is mock_source

    def test_connection_without_table_raises_error(self):
        """Config with 'connection' but no 'table' raises error."""
        with pytest.raises(DataSourceError, match="requires 'table' or 'query'"):
            create_datasource_from_config({"connection": "postgresql://host/db"})

    def test_no_connection_no_type_raises_error(self):
        """Config without 'connection' or 'type' raises error."""
        with pytest.raises(DataSourceError, match="must have either"):
            create_datasource_from_config({"table": "users"})

    @patch("truthound.datasources.sql.get_available_sources")
    def test_type_not_available_raises_error(self, mock_available):
        """Unavailable type raises DataSourceError with available list."""
        mock_available.return_value = {"postgresql": MagicMock(), "mysql": MagicMock()}

        with pytest.raises(DataSourceError, match="not available"):
            create_datasource_from_config({
                "type": "oracle",
                "table": "users",
                "host": "localhost",
            })

    @patch("truthound.datasources.sql.get_available_sources")
    def test_type_style_creates_source(self, mock_available):
        """Config with 'type' constructs from source class."""
        mock_cls = MagicMock()
        mock_source = MagicMock()
        mock_cls.return_value = mock_source
        mock_available.return_value = {"postgresql": mock_cls}

        result = create_datasource_from_config({
            "type": "postgresql",
            "table": "users",
            "host": "localhost",
            "database": "mydb",
        })
        assert result is mock_source
        mock_cls.assert_called_once_with(
            table="users", host="localhost", database="mydb"
        )
