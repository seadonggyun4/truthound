"""Core CLI commands for Truthound.

This package contains the fundamental CLI commands:
    - learn: Learn schema from data files
    - check: Validate data quality
    - scan: Scan for PII
    - mask: Mask sensitive data
    - profile: Generate data profiles
    - compare: Compare datasets for drift
"""

import typer

from truthound.cli_modules.core.learn import learn_cmd
from truthound.cli_modules.core.check import check_cmd
from truthound.cli_modules.core.scan import scan_cmd
from truthound.cli_modules.core.mask import mask_cmd
from truthound.cli_modules.core.profile import profile_cmd
from truthound.cli_modules.core.compare import compare_cmd

# Core app for mounting commands
app = typer.Typer(help="Core data quality commands")


def register_commands(parent_app: typer.Typer) -> None:
    """Register core commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to
    """
    parent_app.command(name="learn")(learn_cmd)
    parent_app.command(name="check")(check_cmd)
    parent_app.command(name="scan")(scan_cmd)
    parent_app.command(name="mask")(mask_cmd)
    parent_app.command(name="profile")(profile_cmd)
    parent_app.command(name="compare")(compare_cmd)


__all__ = [
    "app",
    "register_commands",
    "learn_cmd",
    "check_cmd",
    "scan_cmd",
    "mask_cmd",
    "profile_cmd",
    "compare_cmd",
]
