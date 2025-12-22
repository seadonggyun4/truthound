"""Command-line interface for Truthound."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.api import check, mask, profile, scan
from truthound.drift import compare
from truthound.schema import learn

app = typer.Typer(
    name="truthound",
    help="Zero-configuration data quality toolkit powered by Polars",
    add_completion=False,
)


@app.command(name="learn")
def learn_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file to learn from")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output schema file path"),
    ] = Path("schema.yaml"),
    no_constraints: Annotated[
        bool,
        typer.Option("--no-constraints", help="Don't infer constraints from data"),
    ] = False,
) -> None:
    """Learn schema from a data file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        schema = learn(str(file), infer_constraints=not no_constraints)
        schema.save(output)
        typer.echo(f"Schema saved to {output}")
        typer.echo(f"  Columns: {len(schema.columns)}")
        typer.echo(f"  Rows: {schema.row_count:,}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="check")
def check_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    validators: Annotated[
        Optional[list[str]],
        typer.Option("--validators", "-v", help="Comma-separated list of validators"),
    ] = None,
    min_severity: Annotated[
        Optional[str],
        typer.Option("--min-severity", "-s", help="Minimum severity level (low, medium, high, critical)"),
    ] = None,
    schema_file: Annotated[
        Optional[Path],
        typer.Option("--schema", help="Schema file for validation"),
    ] = None,
    auto_schema: Annotated[
        bool,
        typer.Option("--auto-schema", help="Auto-learn and cache schema (zero-config mode)"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, html)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if issues are found"),
    ] = False,
) -> None:
    """Validate data quality in a file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    if schema_file and not schema_file.exists():
        typer.echo(f"Error: Schema file not found: {schema_file}", err=True)
        raise typer.Exit(1)

    # Parse validators if provided
    validator_list = None
    if validators:
        validator_list = [v.strip() for v in ",".join(validators).split(",")]

    try:
        report = check(
            str(file),
            validators=validator_list,
            min_severity=min_severity,
            schema=schema_file,
            auto_schema=auto_schema,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Output the report
    if format == "json":
        result = report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)
    elif format == "html":
        if not output:
            typer.echo("Error: --output is required for HTML format", err=True)
            raise typer.Exit(1)
        # HTML output requires jinja2
        try:
            from truthound.html_report import generate_html_report

            html = generate_html_report(report)
            output.write_text(html)
            typer.echo(f"HTML report written to {output}")
        except ImportError:
            typer.echo("Error: HTML reports require jinja2. Install with: pip install truthound[reports]", err=True)
            raise typer.Exit(1)
    else:
        report.print()

    # Exit with error if strict mode and issues found
    if strict and report.has_issues:
        raise typer.Exit(1)


@app.command(name="scan")
def scan_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Scan for personally identifiable information (PII)."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        pii_report = scan(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = pii_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)
    else:
        pii_report.print()


@app.command(name="mask")
def mask_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ],
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to mask (comma-separated)"),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Masking strategy (redact, hash, fake)"),
    ] = "redact",
) -> None:
    """Mask sensitive data in a file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    # Parse columns if provided
    column_list = None
    if columns:
        column_list = [c.strip() for c in ",".join(columns).split(",")]

    try:
        masked_df = mask(str(file), columns=column_list, strategy=strategy)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Write output based on file extension
    suffix = output.suffix.lower()
    if suffix == ".csv":
        masked_df.write_csv(output)
    elif suffix == ".parquet":
        masked_df.write_parquet(output)
    elif suffix == ".json":
        masked_df.write_json(output)
    else:
        masked_df.write_csv(output)

    typer.echo(f"Masked data written to {output}")


@app.command(name="profile")
def profile_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Generate a statistical profile of the data."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        profile_report = profile(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = profile_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Profile written to {output}")
        else:
            typer.echo(result)
    else:
        profile_report.print()


@app.command(name="compare")
def compare_cmd(
    baseline: Annotated[Path, typer.Argument(help="Baseline (reference) data file")],
    current: Annotated[Path, typer.Argument(help="Current data file to compare")],
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to compare (comma-separated)"),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Detection method (auto, ks, psi, chi2, js)"),
    ] = "auto",
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", "-t", help="Custom drift threshold"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if drift is detected"),
    ] = False,
) -> None:
    """Compare two datasets and detect data drift."""
    if not baseline.exists():
        typer.echo(f"Error: Baseline file not found: {baseline}", err=True)
        raise typer.Exit(1)

    if not current.exists():
        typer.echo(f"Error: Current file not found: {current}", err=True)
        raise typer.Exit(1)

    # Parse columns if provided
    column_list = None
    if columns:
        column_list = [c.strip() for c in ",".join(columns).split(",")]

    try:
        drift_report = compare(
            str(baseline),
            str(current),
            columns=column_list,
            method=method,
            threshold=threshold,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = drift_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Drift report written to {output}")
        else:
            typer.echo(result)
    else:
        drift_report.print()

    # Exit with error if strict mode and drift found
    if strict and drift_report.has_drift:
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
