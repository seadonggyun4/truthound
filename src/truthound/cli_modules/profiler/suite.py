"""Suite generation commands.

This module implements the `truthound generate-suite` and `truthound quick-suite`
commands for generating validation rules from profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback

# Valid rule categories (must match RuleCategory enum in generators/base.py)
VALID_CATEGORIES = [
    "schema",
    "completeness",
    "uniqueness",
    "format",
    "distribution",
    "pattern",
    "temporal",
    "relationship",
    "anomaly",
]

CATEGORIES_HELP = f"Categories: {', '.join(VALID_CATEGORIES)}"


def _validate_categories(categories: list[str] | None, option_name: str) -> None:
    """Validate that all categories are valid."""
    if not categories:
        return
    invalid = [c for c in categories if c not in VALID_CATEGORIES]
    if invalid:
        raise ValueError(
            f"Invalid {option_name}: {', '.join(invalid)}. "
            f"Valid categories: {', '.join(VALID_CATEGORIES)}"
        )


@error_boundary
def generate_suite_cmd(
    profile_file: Annotated[
        Path,
        typer.Argument(help="Path to profile JSON file (from auto-profile)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (yaml, json, python, toml, checkpoint)",
        ),
    ] = "yaml",
    strictness: Annotated[
        str,
        typer.Option("--strictness", "-s", help="Rule strictness (loose, medium, strict)"),
    ] = "medium",
    include: Annotated[
        Optional[list[str]],
        typer.Option(
            "--include",
            "-i",
            help=f"Include only these categories. {CATEGORIES_HELP}",
        ),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude",
            "-e",
            help=f"Exclude these categories. {CATEGORIES_HELP}",
        ),
    ] = None,
    min_confidence: Annotated[
        Optional[str],
        typer.Option("--min-confidence", help="Minimum rule confidence (low, medium, high)"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name for the validation suite"),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset",
            "-p",
            help="Configuration preset (default, strict, loose, minimal, comprehensive, ci_cd)",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    group_by_category: Annotated[
        bool,
        typer.Option("--group-by-category", help="Group rules by category in output"),
    ] = False,
    code_style: Annotated[
        str,
        typer.Option(
            "--code-style",
            help="Python code style (functional, class_based, declarative)",
        ),
    ] = "functional",
) -> None:
    """Generate validation rules from a profile.

    This creates a validation suite based on the data profile.
    Categories available: schema, completeness, uniqueness, format,
    distribution, pattern, temporal, relationship, anomaly

    Output formats:
        - yaml: Human-readable YAML (default)
        - json: Machine-readable JSON
        - python: Executable Python code
        - toml: TOML configuration
        - checkpoint: Truthound checkpoint format for CI/CD

    Examples:
        truthound generate-suite profile.json -o rules.yaml
        truthound generate-suite profile.json -i schema -i format
        truthound generate-suite profile.json --preset strict
        truthound generate-suite profile.json -f python --code-style class_based
        truthound generate-suite profile.json -f checkpoint -o ci_rules.yaml
    """
    require_file(profile_file, "Profile file")

    try:
        from truthound.profiler import (
            run_generate_suite,
            get_available_formats,
            get_available_presets,
        )

        # Validate format
        available_formats = get_available_formats()
        if format not in available_formats:
            typer.echo(
                f"Error: Invalid format '{format}'. "
                f"Available: {', '.join(available_formats)}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate preset
        if preset:
            available_presets = get_available_presets()
            if preset not in available_presets:
                typer.echo(
                    f"Error: Invalid preset '{preset}'. "
                    f"Available: {', '.join(available_presets)}",
                    err=True,
                )
                raise typer.Exit(1)

        # Parse and validate categories
        include_cats = parse_list_callback(include) if include else None
        exclude_cats = parse_list_callback(exclude) if exclude else None

        try:
            _validate_categories(include_cats, "--include categories")
            _validate_categories(exclude_cats, "--exclude categories")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Run generation using the handler
        exit_code = run_generate_suite(
            profile_file=profile_file,
            output=output,
            format=format,
            strictness=strictness,
            include=include_cats,
            exclude=exclude_cats,
            min_confidence=min_confidence,
            name=name,
            preset=preset,
            config=config,
            group_by_category=group_by_category,
            echo=typer.echo,
            verbose=True,
        )

        if exit_code != 0:
            raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@error_boundary
def quick_suite_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (yaml, json, python, toml, checkpoint)",
        ),
    ] = "yaml",
    strictness: Annotated[
        str,
        typer.Option("--strictness", "-s", help="Rule strictness (loose, medium, strict)"),
    ] = "medium",
    include: Annotated[
        Optional[list[str]],
        typer.Option(
            "--include",
            "-i",
            help=f"Include only these categories. {CATEGORIES_HELP}",
        ),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude",
            "-e",
            help=f"Exclude these categories. {CATEGORIES_HELP}",
        ),
    ] = None,
    min_confidence: Annotated[
        Optional[str],
        typer.Option("--min-confidence", help="Minimum rule confidence (low, medium, high)"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name for the validation suite"),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset",
            "-p",
            help="Configuration preset (default, strict, loose, minimal, comprehensive, ci_cd)",
        ),
    ] = None,
    sample_size: Annotated[
        Optional[int],
        typer.Option("--sample-size", help="Sample size for profiling (default: auto)"),
    ] = None,
) -> None:
    """Profile data and generate validation rules in one step.

    This is a convenience command that combines auto-profile and generate-suite.

    Output formats:
        - yaml: Human-readable YAML (default)
        - json: Machine-readable JSON
        - python: Executable Python code
        - toml: TOML configuration
        - checkpoint: Truthound checkpoint format for CI/CD

    Examples:
        truthound quick-suite data.parquet -o rules.yaml
        truthound quick-suite data.csv -s strict -f python -o validators.py
        truthound quick-suite data.parquet --preset ci_cd -o ci_rules.yaml
        truthound quick-suite large_data.parquet --sample-size 10000
    """
    require_file(file)

    try:
        from truthound.profiler import (
            run_quick_suite,
            get_available_formats,
            get_available_presets,
        )

        # Validate format
        available_formats = get_available_formats()
        if format not in available_formats:
            typer.echo(
                f"Error: Invalid format '{format}'. "
                f"Available: {', '.join(available_formats)}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate preset
        if preset:
            available_presets = get_available_presets()
            if preset not in available_presets:
                typer.echo(
                    f"Error: Invalid preset '{preset}'. "
                    f"Available: {', '.join(available_presets)}",
                    err=True,
                )
                raise typer.Exit(1)

        # Parse and validate categories
        include_cats = parse_list_callback(include) if include else None
        exclude_cats = parse_list_callback(exclude) if exclude else None

        try:
            _validate_categories(include_cats, "--include categories")
            _validate_categories(exclude_cats, "--exclude categories")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Run quick suite using the handler
        exit_code = run_quick_suite(
            file=file,
            output=output,
            format=format,
            strictness=strictness,
            include=include_cats,
            exclude=exclude_cats,
            min_confidence=min_confidence,
            name=name,
            preset=preset,
            sample_size=sample_size,
            echo=typer.echo,
            verbose=True,
        )

        if exit_code != 0:
            raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
