"""Shared DataSource resolution for CLI commands.

This module provides a unified abstraction layer that resolves CLI
options (file path, connection string, or config file) into either
a file path string or a BaseDataSource instance. All core CLI commands
use this layer for consistent data source handling.

Architecture:
    CLI options → resolve_datasource() → (file_path | None, source | None)
                                            ↓               ↓
                                        api.func(data=...)  api.func(source=...)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import typer

from truthound.cli_modules.common.errors import (
    CLIError,
    DataSourceError,
    ErrorCode,
    FileNotFoundError,
    require_file,
)

if TYPE_CHECKING:
    from truthound.datasources.base import BaseDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Reusable Annotated CLI Options
# =============================================================================

ConnectionOpt = Annotated[
    Optional[str],
    typer.Option(
        "--connection",
        "--conn",
        help=(
            "Database connection string. "
            "Examples: postgresql://user:pass@host:5432/db, "
            "mysql://user:pass@host/db, sqlite:///path/to.db"
        ),
    ),
]

TableOpt = Annotated[
    Optional[str],
    typer.Option(
        "--table",
        help="Database table name (required with --connection for SQL sources)",
    ),
]

QueryOpt = Annotated[
    Optional[str],
    typer.Option(
        "--query",
        help="SQL query to validate (alternative to --table)",
    ),
]

SourceConfigOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--source-config",
        "--sc",
        help=(
            "Path to data source configuration file (JSON/YAML). "
            "See docs for config file format."
        ),
    ),
]

SourceNameOpt = Annotated[
    Optional[str],
    typer.Option(
        "--source-name",
        help="Custom name for the data source (used in report labels)",
    ),
]


# =============================================================================
# DataSource Resolution
# =============================================================================


def resolve_datasource(
    file: Path | None = None,
    connection: str | None = None,
    table: str | None = None,
    query: str | None = None,
    source_config: Path | None = None,
    source_name: str | None = None,
) -> tuple[str | None, "BaseDataSource | None"]:
    """Resolve CLI options into a file path or BaseDataSource instance.

    This is the central resolution function used by all CLI commands.
    It enforces mutual exclusivity between input modes and validates
    required parameters for each mode.

    Args:
        file: Path to a data file (CSV, JSON, Parquet, etc.)
        connection: Database connection string
        table: Database table name (for SQL sources)
        query: SQL query string (alternative to table)
        source_config: Path to a JSON/YAML data source config file
        source_name: Custom label for the data source

    Returns:
        A tuple of (file_path, source) where exactly one is non-None.

    Raises:
        DataSourceError: If inputs are invalid or conflicting.
        FileNotFoundError: If the specified file does not exist.
    """
    _validate_input_exclusivity(file, connection, source_config)

    # Mode 1: Source config file
    if source_config is not None:
        require_file(source_config, "Source config file")
        config = parse_source_config(source_config)
        source = create_datasource_from_config(config)
        if source_name:
            _set_source_name(source, source_name)
        return None, source

    # Mode 2: Connection string
    if connection is not None:
        source = _create_from_connection(connection, table, query, source_name)
        return None, source

    # Mode 3: File path (legacy, default)
    if file is not None:
        require_file(file)
        return str(file), None

    # No input provided
    raise DataSourceError(
        "No data input specified.",
        hint=(
            "Provide one of:\n"
            "  - A file path:       truthound <command> data.csv\n"
            "  - A connection:      truthound <command> --connection 'postgresql://...' --table users\n"
            "  - A config file:     truthound <command> --source-config db.yaml"
        ),
    )


def resolve_compare_sources(
    baseline: Path | None = None,
    current: Path | None = None,
    source_config: Path | None = None,
) -> tuple[
    tuple[str | None, "BaseDataSource | None"],
    tuple[str | None, "BaseDataSource | None"],
]:
    """Resolve inputs for the compare command (dual-source).

    Args:
        baseline: Baseline file path
        current: Current file path
        source_config: Config file with baseline/current sections

    Returns:
        Tuple of (baseline_resolution, current_resolution),
        each a (file_path | None, source | None) pair.

    Raises:
        DataSourceError: If inputs are invalid or conflicting.
    """
    if source_config is not None:
        if baseline is not None or current is not None:
            raise DataSourceError(
                "Cannot specify both file paths and --source-config for compare.",
                hint="Use either positional file args OR --source-config, not both.",
            )
        require_file(source_config, "Source config file")
        config = parse_source_config(source_config)

        baseline_cfg = config.get("baseline")
        current_cfg = config.get("current")
        if not baseline_cfg or not current_cfg:
            raise DataSourceError(
                "Compare source config must have 'baseline' and 'current' sections.",
                hint=(
                    "Example config:\n"
                    "  baseline:\n"
                    "    connection: postgresql://...\n"
                    "    table: train_data\n"
                    "  current:\n"
                    "    connection: postgresql://...\n"
                    "    table: prod_data"
                ),
            )

        baseline_source = create_datasource_from_config(baseline_cfg)
        current_source = create_datasource_from_config(current_cfg)
        return (None, baseline_source), (None, current_source)

    # File-based path
    if baseline is None or current is None:
        raise DataSourceError(
            "Both baseline and current data must be specified.",
            hint=(
                "Provide two file paths:\n"
                "  truthound compare baseline.csv current.csv\n"
                "Or use --source-config with baseline/current sections."
            ),
        )

    require_file(baseline, "Baseline file")
    require_file(current, "Current file")
    return (str(baseline), None), (str(current), None)


# =============================================================================
# Config File Parsing
# =============================================================================


def parse_source_config(config_path: Path) -> dict[str, Any]:
    """Parse a data source configuration file (JSON or YAML).

    Supported formats:
        - JSON (.json)
        - YAML (.yaml, .yml)

    Config schema for single source:
        type: postgresql
        connection: "postgresql://user:pass@host:5432/db"
        table: users

    Config schema for compare (dual source):
        baseline:
            connection: "postgresql://..."
            table: train_data
        current:
            connection: "postgresql://..."
            table: prod_data

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        DataSourceError: If the file cannot be parsed.
    """
    content = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        try:
            config = json.loads(content)
        except json.JSONDecodeError as e:
            raise DataSourceError(
                f"Invalid JSON in source config: {e}",
                hint=f"Check the syntax of {config_path}",
            )
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml

            config = yaml.safe_load(content)
        except ImportError:
            raise DataSourceError(
                "YAML config requires PyYAML.",
                hint="Install with: pip install pyyaml",
            )
        except Exception as e:
            raise DataSourceError(
                f"Invalid YAML in source config: {e}",
                hint=f"Check the syntax of {config_path}",
            )
    else:
        raise DataSourceError(
            f"Unsupported config file format: {suffix}",
            hint="Use .json, .yaml, or .yml",
        )

    if not isinstance(config, dict):
        raise DataSourceError(
            "Source config must be a JSON/YAML object (dictionary).",
            hint=f"Check {config_path}",
        )

    return config


def create_datasource_from_config(config: dict[str, Any]) -> "BaseDataSource":
    """Create a BaseDataSource from a parsed configuration dictionary.

    Supports two config styles:
        1. Connection string style:
            {"connection": "postgresql://...", "table": "users"}

        2. Individual parameters style:
            {"type": "postgresql", "host": "localhost", "database": "mydb",
             "user": "postgres", "password": "...", "table": "users"}

    Args:
        config: Configuration dictionary.

    Returns:
        Configured BaseDataSource instance.

    Raises:
        DataSourceError: If the config is invalid or the backend is unavailable.
    """
    from truthound.datasources.factory import get_sql_datasource
    from truthound.datasources.sql import get_available_sources

    connection = config.get("connection")
    table = config.get("table")
    query = config.get("query")
    source_type = config.get("type")

    # Style 1: Connection string
    if connection:
        if not table and not query:
            raise DataSourceError(
                "Config with 'connection' requires 'table' or 'query'.",
                hint="Add a 'table' or 'query' field to your config file.",
            )
        try:
            return get_sql_datasource(
                connection, table=table or "__query__", query=query
            )
        except Exception as e:
            raise DataSourceError(
                f"Failed to create data source from connection string: {e}",
                source_type=source_type,
            )

    # Style 2: Individual parameters with type
    if not source_type:
        raise DataSourceError(
            "Config must have either 'connection' or 'type' field.",
            hint=(
                "Example:\n"
                "  connection: postgresql://user:pass@host:5432/db\n"
                "  table: users\n"
                "Or:\n"
                "  type: postgresql\n"
                "  host: localhost\n"
                "  database: mydb\n"
                "  table: users"
            ),
        )

    if not table and not query:
        raise DataSourceError(
            f"Config for type '{source_type}' requires 'table' or 'query'.",
        )

    available = get_available_sources()
    source_cls = available.get(source_type)
    if source_cls is None:
        available_names = [k for k, v in available.items() if v is not None]
        raise DataSourceError(
            f"Data source type '{source_type}' is not available.",
            source_type=source_type,
            hint=(
                f"Available types: {', '.join(available_names)}. "
                f"You may need to install the required driver."
            ),
        )

    # Build constructor kwargs from config (exclude meta keys)
    meta_keys = {"type", "table", "query", "name"}
    kwargs: dict[str, Any] = {}
    if table:
        kwargs["table"] = table
    if query:
        kwargs["query"] = query
    for key, value in config.items():
        if key not in meta_keys:
            kwargs[key] = value

    try:
        return source_cls(**kwargs)
    except TypeError as e:
        raise DataSourceError(
            f"Invalid config for '{source_type}': {e}",
            source_type=source_type,
            hint=f"Check the supported parameters for {source_type} data source.",
        )
    except Exception as e:
        raise DataSourceError(
            f"Failed to create '{source_type}' data source: {e}",
            source_type=source_type,
        )


# =============================================================================
# Internal Helpers
# =============================================================================


def _validate_input_exclusivity(
    file: Path | None,
    connection: str | None,
    source_config: Path | None,
) -> None:
    """Validate that at most one data input mode is specified."""
    modes = []
    if file is not None:
        modes.append("file argument")
    if connection is not None:
        modes.append("--connection")
    if source_config is not None:
        modes.append("--source-config")

    if len(modes) > 1:
        raise DataSourceError(
            f"Conflicting data inputs: {' and '.join(modes)}.",
            hint="Specify only one: a file path, --connection, or --source-config.",
        )


def _create_from_connection(
    connection: str,
    table: str | None,
    query: str | None,
    source_name: str | None,
) -> "BaseDataSource":
    """Create a BaseDataSource from a connection string."""
    from truthound.datasources.factory import get_sql_datasource

    if not table and not query:
        raise DataSourceError(
            "--table or --query is required with --connection.",
            hint=(
                "Example:\n"
                "  --connection 'postgresql://user:pass@host/db' --table users\n"
                "  --connection 'sqlite:///data.db' --query 'SELECT * FROM orders'"
            ),
        )

    try:
        target = table or "__query__"
        source = get_sql_datasource(connection, table=target, query=query)
    except ImportError as e:
        _raise_driver_hint(connection, e)
    except Exception as e:
        raise DataSourceError(
            f"Failed to connect: {e}",
            hint="Check the connection string format and database availability.",
        )

    if source_name:
        _set_source_name(source, source_name)

    return source


def _set_source_name(source: "BaseDataSource", name: str) -> None:
    """Attempt to set a custom name on a data source."""
    if hasattr(source, "config") and hasattr(source.config, "name"):
        try:
            source.config.name = name
        except (AttributeError, TypeError):
            pass


def _raise_driver_hint(connection: str, error: ImportError) -> None:
    """Raise a DataSourceError with install hints based on connection string."""
    conn_lower = connection.lower()
    hints = {
        "postgresql": ("psycopg2-binary", "pip install truthound[postgresql]"),
        "postgres": ("psycopg2-binary", "pip install truthound[postgresql]"),
        "mysql": ("pymysql", "pip install truthound[mysql]"),
        "oracle": ("oracledb", "pip install oracledb"),
        "mssql": ("pyodbc", "pip install pyodbc"),
        "sqlserver": ("pyodbc", "pip install pyodbc"),
        "bigquery": ("google-cloud-bigquery", "pip install truthound[bigquery]"),
        "snowflake": ("snowflake-connector-python", "pip install truthound[snowflake]"),
        "redshift": ("redshift-connector", "pip install truthound[redshift]"),
        "databricks": ("databricks-sql-connector", "pip install truthound[databricks]"),
        "duckdb": ("duckdb", "pip install duckdb"),
    }

    for prefix, (pkg, install_cmd) in hints.items():
        if prefix in conn_lower:
            raise DataSourceError(
                f"Missing driver for {prefix}: {error}",
                source_type=prefix,
                hint=f"Install with: {install_cmd}",
            )

    raise DataSourceError(
        f"Missing driver: {error}",
        hint="Check that the required database driver is installed.",
    )
