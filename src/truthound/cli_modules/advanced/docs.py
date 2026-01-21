"""Data documentation commands.

This module implements commands for generating data documentation
and reports (Phase 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file

# Docs app for subcommands
app = typer.Typer(
    name="docs",
    help="Generate data documentation and reports (Phase 8)",
)


# Known data file extensions that are NOT profile JSON
_DATA_FILE_EXTENSIONS = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".feather", ".arrow"}


@app.command(name="generate")
@error_boundary
def generate_cmd(
    profile_file: Annotated[
        Path,
        typer.Argument(help="Path to profile JSON file (from auto-profile)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Report title"),
    ] = "Data Profile Report",
    subtitle: Annotated[
        str,
        typer.Option("--subtitle", "-s", help="Report subtitle"),
    ] = "",
    theme: Annotated[
        str,
        typer.Option("--theme", help="Report theme (light, dark, professional, minimal, modern)"),
    ] = "professional",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (html, pdf)"),
    ] = "html",
) -> None:
    """Generate HTML report from profile data.

    This creates a static, self-contained HTML report that can be:
    - Saved as CI/CD artifact
    - Shared via email or Slack
    - Viewed offline in any browser

    Charts are rendered using ApexCharts (interactive, feature-rich).
    PDF output uses SVG for best compatibility.

    Examples:
        truthound docs generate profile.json -o report.html
        truthound docs generate profile.json -o report.html --title "Q4 Data Report" --theme dark
        truthound docs generate profile.json -o report.pdf --format pdf
    """
    require_file(profile_file, "Profile file")

    # Check if user passed a data file instead of profile JSON
    if profile_file.suffix.lower() in _DATA_FILE_EXTENSIONS:
        typer.echo(
            f"Error: '{profile_file}' appears to be a data file, not a profile JSON.\n\n"
            f"This command requires a profile JSON file from 'auto-profile'.\n\n"
            f"To generate a report from your data:\n"
            f"  1. First, create a profile:\n"
            f"     truthound auto-profile {profile_file} -o profile.json\n\n"
            f"  2. Then, generate the report:\n"
            f"     truthound docs generate profile.json -o report.html",
            err=True,
        )
        raise typer.Exit(1)

    # Default output path
    if not output:
        output = profile_file.with_suffix(f".{format}")

    try:
        from truthound.datadocs import (
            generate_html_report,
            export_to_pdf,
        )

        # Load profile
        with open(profile_file, "r", encoding="utf-8") as f:
            try:
                profile = json.load(f)
            except json.JSONDecodeError as e:
                # Check if file extension suggests it might be a data file
                typer.echo(
                    f"Error: Failed to parse '{profile_file}' as JSON.\n\n"
                    f"This command requires a profile JSON file from 'auto-profile'.\n"
                    f"JSON parse error: {e}\n\n"
                    f"If you want to generate a report from a data file:\n"
                    f"  1. First, create a profile:\n"
                    f"     truthound auto-profile <your-data-file> -o profile.json\n\n"
                    f"  2. Then, generate the report:\n"
                    f"     truthound docs generate profile.json -o report.html",
                    err=True,
                )
                raise typer.Exit(1)

        typer.echo(f"Generating {format.upper()} report...")
        typer.echo(f"  Profile: {profile_file}")
        typer.echo(f"  Theme: {theme}")

        if format == "html":
            html_content = generate_html_report(
                profile=profile,
                title=title,
                subtitle=subtitle,
                theme=theme,
                output_path=output,
            )
            typer.echo(f"\nReport saved to: {output}")
            typer.echo(f"  Size: {len(html_content):,} bytes")

        elif format == "pdf":
            try:
                output_path = export_to_pdf(
                    profile=profile,
                    output_path=output,
                    title=title,
                    subtitle=subtitle,
                    theme=theme,
                )
                typer.echo(f"\nPDF report saved to: {output_path}")
            except ImportError as e:
                # WeasyPrintDependencyError includes detailed installation instructions
                typer.echo(f"\nError: {e}", err=True)
                raise typer.Exit(1)
            except OSError as e:
                # Catch system library errors that weren't caught in export_to_pdf
                if "cannot load library" in str(e):
                    from truthound.datadocs.builder import _get_weasyprint_install_instructions
                    typer.echo(f"\nError: {e}\n\n{_get_weasyprint_install_instructions()}", err=True)
                else:
                    typer.echo(f"\nError: {e}", err=True)
                raise typer.Exit(1)

        else:
            typer.echo(f"Error: Unsupported format '{format}'", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="themes")
@error_boundary
def themes_cmd() -> None:
    """List available report themes.

    Displays all supported themes with descriptions.
    """
    try:
        from truthound.datadocs import get_available_themes

        typer.echo("Available report themes:")
        typer.echo("")

        themes_info = {
            "light": "Clean and bright, suitable for most use cases",
            "dark": "Dark mode with vibrant colors, easy on the eyes",
            "professional": "Corporate style, subdued colors (default)",
            "minimal": "Minimalist design with monochrome accents",
            "modern": "Contemporary design with vibrant gradients",
        }

        for theme in get_available_themes():
            desc = themes_info.get(theme, "")
            typer.echo(f"  {theme:14} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@error_boundary
def dashboard_cmd(
    profile: Annotated[
        Optional[Path],
        typer.Option("--profile", "-p", help="Path to profile JSON file"),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", help="Server port"),
    ] = 8080,
    host: Annotated[
        str,
        typer.Option("--host", help="Server host"),
    ] = "localhost",
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Dashboard title"),
    ] = "Truthound Dashboard",
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug mode"),
    ] = False,
) -> None:
    """Launch interactive dashboard for data exploration.

    This requires the dashboard extra to be installed:
        pip install truthound[dashboard]

    The dashboard provides:
    - Interactive data exploration
    - Column filtering and search
    - Real-time quality metrics
    - Pattern visualization

    Examples:
        truthound dashboard --profile profile.json
        truthound dashboard --profile profile.json --port 3000 --title "My Dashboard"
    """
    try:
        from truthound.datadocs import launch_dashboard

        if profile:
            require_file(profile, "Profile file")

        typer.echo(f"Launching dashboard on http://{host}:{port}")
        if profile:
            typer.echo(f"  Profile: {profile}")

        launch_dashboard(
            profile_path=profile,
            port=port,
            host=host,
            title=title,
            debug=debug,
        )

    except ImportError:
        typer.echo(
            "Error: Dashboard requires additional dependencies. "
            "Install with: pip install truthound[dashboard]",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
