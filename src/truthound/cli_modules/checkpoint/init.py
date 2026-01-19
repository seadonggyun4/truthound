"""Checkpoint init command.

This module implements the `truthound checkpoint init` command for
initializing sample checkpoint configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from truthound.cli_modules.common.errors import error_boundary


# Sample configuration template
SAMPLE_CONFIG = {
    "checkpoints": [
        {
            "name": "daily_data_validation",
            "data_source": "data/production.csv",
            "validators": ["null", "duplicate", "range", "regex"],
            "validator_config": {
                "regex": {
                    "patterns": {
                        # Email validation pattern
                        "email": r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
                        # Product code pattern (e.g., PROD-001, SKU_12345)
                        "product_code": r"^[A-Z]{2,4}[-_][0-9]{3,6}$",
                        # Phone number pattern (optional country code)
                        "phone": r"^(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
                    }
                },
                "range": {
                    # Column-specific range constraints
                    "columns": {
                        "age": {"min_value": 0, "max_value": 150},
                        "price": {"min_value": 0},
                    }
                },
            },
            "min_severity": "medium",
            "auto_schema": True,
            "tags": {
                "environment": "production",
                "team": "data-platform",
            },
            "actions": [
                {
                    "type": "store_result",
                    "store_path": "./truthound_results",
                    "partition_by": "date",
                },
                {
                    "type": "update_docs",
                    "site_path": "./truthound_docs",
                    "include_history": True,
                },
                {
                    "type": "slack",
                    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                    "notify_on": "failure",
                    "channel": "#data-quality",
                },
            ],
            "triggers": [
                {
                    "type": "schedule",
                    "interval_hours": 24,
                    "run_on_weekdays": [0, 1, 2, 3, 4],
                },
            ],
        },
        {
            "name": "hourly_metrics_check",
            "data_source": "data/metrics.parquet",
            "validators": ["null", "range"],
            "validator_config": {
                "range": {
                    "columns": {
                        "value": {"min_value": 0, "max_value": 100},
                        "count": {"min_value": 0},
                    }
                },
            },
            "actions": [
                {
                    "type": "webhook",
                    "url": "https://api.example.com/data-quality/events",
                    "auth_type": "bearer",
                    "auth_credentials": {"token": "${API_TOKEN}"},
                },
            ],
            "triggers": [
                {
                    "type": "cron",
                    "expression": "0 * * * *",
                },
            ],
        },
    ],
}


@error_boundary
def init_cmd(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("truthound.yaml"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Config format (yaml, json)"),
    ] = "yaml",
) -> None:
    """Initialize a sample checkpoint configuration file.

    This command creates a starter configuration file with example
    checkpoints that you can customize for your needs.

    Examples:
        truthound checkpoint init
        truthound checkpoint init -o my_config.yaml
        truthound checkpoint init --format json -o config.json
    """
    import yaml

    if format == "json":
        output = output.with_suffix(".json")
        output.write_text(json.dumps(SAMPLE_CONFIG, indent=2))
    else:
        output = output.with_suffix(".yaml")
        output.write_text(
            yaml.dump(SAMPLE_CONFIG, default_flow_style=False, sort_keys=False)
        )

    typer.echo(f"Sample checkpoint config created: {output}")
    typer.echo("\nEdit the file to configure your checkpoints, then run:")
    typer.echo(f"  truthound checkpoint run <checkpoint_name> --config {output}")
