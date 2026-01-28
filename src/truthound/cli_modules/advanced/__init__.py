"""Advanced CLI commands for Truthound.

This package contains advanced feature commands:
    - docs: Data documentation and reports
    - ml: Machine learning based validation
    - lineage: Data lineage tracking
    - realtime: Real-time streaming validation
    - benchmark: Performance benchmarking
    - quality: Quality score reporting and filtering
"""

import typer

# Import submodule apps
from truthound.cli_modules.advanced import docs
from truthound.cli_modules.advanced import ml
from truthound.cli_modules.advanced import lineage
from truthound.cli_modules.advanced import realtime
from truthound.cli_modules.advanced import benchmark
from truthound.cli_modules.advanced import quality


def register_commands(parent_app: typer.Typer) -> None:
    """Register advanced commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to
    """
    # Register sub-apps
    parent_app.add_typer(docs.app, name="docs")
    parent_app.add_typer(ml.app, name="ml")
    parent_app.add_typer(lineage.app, name="lineage")
    parent_app.add_typer(realtime.app, name="realtime")
    parent_app.add_typer(benchmark.app, name="benchmark")
    parent_app.add_typer(quality.app, name="quality")

    # Register top-level dashboard command
    parent_app.command(name="dashboard")(docs.dashboard_cmd)


__all__ = [
    "register_commands",
    "docs",
    "ml",
    "lineage",
    "realtime",
    "benchmark",
    "quality",
]
