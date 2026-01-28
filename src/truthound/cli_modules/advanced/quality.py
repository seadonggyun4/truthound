"""CLI commands for Quality Reporter.

This module provides CLI commands for generating quality reports,
filtering quality scores, and comparing rules by quality metrics.

Commands:
    th quality report: Generate quality reports in various formats
    th quality filter: Filter quality scores by criteria
    th quality compare: Compare and rank rules by quality
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from truthound.cli_modules.common.errors import CLIError, error_boundary
from truthound.cli_modules.common.output import ConsoleOutput

# Create app
app = typer.Typer(
    name="quality",
    help="Quality score reporting and filtering commands.",
    no_args_is_help=True,
)

console = Console()
output = ConsoleOutput()


# =============================================================================
# Type Aliases
# =============================================================================

FormatOpt = Annotated[
    str,
    typer.Option(
        "--format", "-f",
        help="Output format (console, json, html, markdown, junit)",
    ),
]

OutputOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--output", "-o",
        help="Output file path",
    ),
]

MinLevelOpt = Annotated[
    Optional[str],
    typer.Option(
        "--min-level",
        help="Minimum quality level (excellent, good, acceptable, poor, unacceptable)",
    ),
]

MinF1Opt = Annotated[
    Optional[float],
    typer.Option(
        "--min-f1",
        help="Minimum F1 score (0.0-1.0)",
    ),
]

ShouldUseOpt = Annotated[
    bool,
    typer.Option(
        "--should-use-only",
        help="Only include rules that should be used",
    ),
]

MaxScoresOpt = Annotated[
    Optional[int],
    typer.Option(
        "--max", "-n",
        help="Maximum number of scores to include",
    ),
]

SortByOpt = Annotated[
    str,
    typer.Option(
        "--sort-by",
        help="Sort by metric (f1, precision, recall, confidence, name)",
    ),
]

DescendingOpt = Annotated[
    bool,
    typer.Option(
        "--desc/--asc",
        help="Sort in descending order",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================


def load_scores_from_file(file_path: Path) -> list:
    """Load quality scores from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of RuleQualityScore objects.

    Raises:
        CLIError: If file cannot be loaded.
    """
    try:
        with open(file_path) as f:
            data = json.load(f)

        # Import here to avoid circular imports
        from truthound.profiler.quality import (
            QualityMetrics,
            QualityLevel,
            RuleType,
            RuleQualityScore,
            ConfusionMatrix,
        )

        scores = []
        items = data if isinstance(data, list) else data.get("scores", [])

        for item in items:
            metrics_data = item.get("metrics", {})

            # Build confusion matrix if present
            cm = None
            if "confusion_matrix" in metrics_data:
                cm_data = metrics_data["confusion_matrix"]
                cm = ConfusionMatrix(
                    true_positives=cm_data.get("true_positives", 0),
                    true_negatives=cm_data.get("true_negatives", 0),
                    false_positives=cm_data.get("false_positives", 0),
                    false_negatives=cm_data.get("false_negatives", 0),
                )

            # Build metrics
            metrics = QualityMetrics(
                precision=metrics_data.get("precision", 0.0),
                recall=metrics_data.get("recall", 0.0),
                f1_score=metrics_data.get("f1_score", 0.0),
                accuracy=metrics_data.get("accuracy", 0.0),
                specificity=metrics_data.get("specificity", 0.0),
                mcc=metrics_data.get("mcc", 0.0),
                confidence=metrics_data.get("confidence", 0.0),
                quality_level=QualityLevel(metrics_data.get("quality_level", "unacceptable")),
                sample_size=metrics_data.get("sample_size", 0),
                population_size=metrics_data.get("population_size", 0),
                confusion_matrix=cm,
            )

            # Build score
            score = RuleQualityScore(
                rule_name=item.get("rule_name", "unknown"),
                rule_type=RuleType(item.get("rule_type", "custom")),
                column=item.get("column"),
                metrics=metrics,
                recommendation=item.get("recommendation", ""),
                should_use=item.get("should_use", False),
                alternatives=item.get("alternatives", []),
            )
            scores.append(score)

        return scores

    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON in {file_path}: {e}")
    except FileNotFoundError:
        raise CLIError(f"File not found: {file_path}")
    except Exception as e:
        raise CLIError(f"Failed to load scores from {file_path}: {e}")


def display_scores_table(scores: list, title: str = "Quality Scores") -> None:
    """Display scores in a Rich table.

    Args:
        scores: List of RuleQualityScore objects.
        title: Table title.
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Rule Name", style="cyan", no_wrap=True)
    table.add_column("Level", justify="center")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Use?", justify="center")

    level_colors = {
        "excellent": "green",
        "good": "blue",
        "acceptable": "yellow",
        "poor": "dark_orange",
        "unacceptable": "red",
    }

    for score in scores:
        level = score.metrics.quality_level.value.lower()
        level_color = level_colors.get(level, "white")
        use_icon = "✓" if score.should_use else "✗"
        use_color = "green" if score.should_use else "red"

        table.add_row(
            score.rule_name[:30],
            f"[{level_color}]{level}[/{level_color}]",
            f"{score.metrics.f1_score:.2%}",
            f"{score.metrics.precision:.2%}",
            f"{score.metrics.recall:.2%}",
            f"{score.metrics.confidence:.2%}",
            f"[{use_color}]{use_icon}[/{use_color}]",
        )

    console.print(table)


# =============================================================================
# Report Command
# =============================================================================


@app.command(name="report")
@error_boundary
def report_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file containing quality scores",
            exists=True,
        ),
    ],
    format: FormatOpt = "console",
    output: OutputOpt = None,
    min_level: MinLevelOpt = None,
    min_f1: MinF1Opt = None,
    should_use_only: ShouldUseOpt = False,
    max_scores: MaxScoresOpt = None,
    sort_by: SortByOpt = "f1",
    descending: DescendingOpt = True,
    include_charts: Annotated[
        bool,
        typer.Option("--charts/--no-charts", help="Include charts in HTML output"),
    ] = True,
    title: Annotated[
        Optional[str],
        typer.Option("--title", "-t", help="Report title"),
    ] = None,
) -> None:
    """Generate a quality report from scored rules.

    Load quality scores from a JSON file and generate a report in the
    specified format.

    Examples:

        # Console report
        th quality report scores.json

        # HTML report with charts
        th quality report scores.json -f html -o report.html

        # JSON report filtered by level
        th quality report scores.json -f json --min-level good

        # Markdown report for documentation
        th quality report scores.json -f markdown -o QUALITY.md
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading scores...", total=None)

        # Load scores
        scores = load_scores_from_file(input_file)

        if not scores:
            console.print("[yellow]No quality scores found in input file.[/yellow]")
            return

        progress.add_task("Generating report...", total=None)

        # Import reporting components
        from truthound.reporters.quality import (
            get_quality_reporter,
            QualityFilter,
            QualityReporterConfig,
        )
        from truthound.reporters.quality.config import ReportSortOrder

        # Build filter
        filter_obj = None
        if min_level or min_f1 is not None or should_use_only:
            filters = []
            if min_level:
                filters.append(QualityFilter.by_level(min_level=min_level))
            if min_f1 is not None:
                filters.append(QualityFilter.by_metric("f1_score", ">=", min_f1))
            if should_use_only:
                filters.append(QualityFilter.by_recommendation(should_use=True))

            if len(filters) == 1:
                filter_obj = filters[0]
            else:
                filter_obj = QualityFilter.all_of(*filters)

            scores = filter_obj.apply(scores)

        # Determine sort order
        sort_map = {
            "f1": ReportSortOrder.F1_DESC if descending else ReportSortOrder.F1_ASC,
            "f1_score": ReportSortOrder.F1_DESC if descending else ReportSortOrder.F1_ASC,
            "precision": ReportSortOrder.PRECISION_DESC if descending else ReportSortOrder.PRECISION_ASC,
            "recall": ReportSortOrder.RECALL_DESC if descending else ReportSortOrder.RECALL_ASC,
            "confidence": ReportSortOrder.CONFIDENCE_DESC if descending else ReportSortOrder.CONFIDENCE_ASC,
            "name": ReportSortOrder.NAME_DESC if descending else ReportSortOrder.NAME_ASC,
        }
        sort_order = sort_map.get(sort_by.lower(), ReportSortOrder.F1_DESC)

        # Create config
        config = QualityReporterConfig(
            title=title or "Quality Score Report",
            sort_order=sort_order,
            max_scores=max_scores,
            include_charts=include_charts,
        )

        # Generate report
        reporter = get_quality_reporter(format, config=config)
        content = reporter.render(scores)

        # Output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            console.print(f"[green]Report written to {output}[/green]")
        else:
            if format == "console":
                console.print(content)
            else:
                print(content)


# =============================================================================
# Filter Command
# =============================================================================


@app.command(name="filter")
@error_boundary
def filter_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file containing quality scores",
            exists=True,
        ),
    ],
    output: OutputOpt = None,
    min_level: MinLevelOpt = None,
    max_level: Annotated[
        Optional[str],
        typer.Option("--max-level", help="Maximum quality level"),
    ] = None,
    min_f1: MinF1Opt = None,
    max_f1: Annotated[
        Optional[float],
        typer.Option("--max-f1", help="Maximum F1 score"),
    ] = None,
    min_precision: Annotated[
        Optional[float],
        typer.Option("--min-precision", help="Minimum precision"),
    ] = None,
    min_recall: Annotated[
        Optional[float],
        typer.Option("--min-recall", help="Minimum recall"),
    ] = None,
    min_confidence: Annotated[
        Optional[float],
        typer.Option("--min-confidence", help="Minimum confidence"),
    ] = None,
    should_use_only: ShouldUseOpt = False,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", "-c", help="Comma-separated list of columns to include"),
    ] = None,
    rule_types: Annotated[
        Optional[str],
        typer.Option("--rule-types", help="Comma-separated list of rule types to include"),
    ] = None,
    invert: Annotated[
        bool,
        typer.Option("--invert", "-v", help="Invert filter (show non-matching)"),
    ] = False,
) -> None:
    """Filter quality scores by various criteria.

    Apply filters to quality scores and output the matching (or non-matching
    if --invert is used) scores.

    Examples:

        # Filter by minimum level
        th quality filter scores.json --min-level good

        # Filter by F1 score range
        th quality filter scores.json --min-f1 0.7 --max-f1 0.95

        # Filter by specific columns
        th quality filter scores.json --columns email,phone,name

        # Filter and save to file
        th quality filter scores.json --min-level acceptable -o filtered.json
    """
    # Load scores
    scores = load_scores_from_file(input_file)

    if not scores:
        console.print("[yellow]No quality scores found in input file.[/yellow]")
        return

    original_count = len(scores)

    # Import filter components
    from truthound.reporters.quality.filters import QualityFilter

    # Build filter
    filters = []

    if min_level or max_level:
        filters.append(QualityFilter.by_level(min_level=min_level, max_level=max_level))

    if min_f1 is not None:
        filters.append(QualityFilter.by_metric("f1_score", ">=", min_f1))
    if max_f1 is not None:
        filters.append(QualityFilter.by_metric("f1_score", "<=", max_f1))

    if min_precision is not None:
        filters.append(QualityFilter.by_metric("precision", ">=", min_precision))
    if min_recall is not None:
        filters.append(QualityFilter.by_metric("recall", ">=", min_recall))
    if min_confidence is not None:
        filters.append(QualityFilter.by_confidence(min_value=min_confidence))

    if should_use_only:
        filters.append(QualityFilter.by_recommendation(should_use=True))

    if columns:
        col_list = [c.strip() for c in columns.split(",")]
        filters.append(QualityFilter.by_column(include=col_list))

    if rule_types:
        type_list = [t.strip() for t in rule_types.split(",")]
        filters.append(QualityFilter.by_rule_type(include=type_list))

    # Combine filters
    if filters:
        combined = QualityFilter.all_of(*filters)
        if invert:
            combined = combined.not_()
        scores = combined.apply(scores)

    # Output
    filtered_count = len(scores)
    console.print(f"[cyan]Filtered: {original_count} → {filtered_count} scores[/cyan]")

    if output:
        # Save as JSON
        data = {"scores": [s.to_dict() for s in scores], "count": len(scores)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Filtered scores written to {output}[/green]")
    else:
        # Display table
        display_scores_table(scores, title=f"Filtered Quality Scores ({filtered_count})")


# =============================================================================
# Compare Command
# =============================================================================


@app.command(name="compare")
@error_boundary
def compare_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file containing quality scores",
            exists=True,
        ),
    ],
    sort_by: SortByOpt = "f1",
    descending: DescendingOpt = True,
    max_scores: MaxScoresOpt = None,
    output: OutputOpt = None,
    group_by: Annotated[
        Optional[str],
        typer.Option("--group-by", "-g", help="Group by: column, rule_type, level"),
    ] = None,
) -> None:
    """Compare and rank rules by quality metrics.

    Load quality scores and display them ranked by the specified metric.
    Optionally group by column, rule type, or quality level.

    Examples:

        # Rank by F1 score (default)
        th quality compare scores.json

        # Rank by precision
        th quality compare scores.json --sort-by precision

        # Top 10 rules
        th quality compare scores.json --max 10

        # Group by column
        th quality compare scores.json --group-by column
    """
    # Load scores
    scores = load_scores_from_file(input_file)

    if not scores:
        console.print("[yellow]No quality scores found in input file.[/yellow]")
        return

    # Import components
    from truthound.reporters.quality.engine import compare_quality_scores

    # Sort scores
    sorted_scores = compare_quality_scores(scores, sort_by=sort_by, descending=descending)

    # Limit
    if max_scores:
        sorted_scores = sorted_scores[:max_scores]

    # Group if requested
    if group_by:
        groups: dict[str, list] = {}
        for score in sorted_scores:
            if group_by == "column":
                key = score.column or "_table_"
            elif group_by == "rule_type":
                key = score.rule_type.value if hasattr(score.rule_type, "value") else str(score.rule_type)
            elif group_by == "level":
                key = score.metrics.quality_level.value
            else:
                key = "all"

            if key not in groups:
                groups[key] = []
            groups[key].append(score)

        for group_name, group_scores in sorted(groups.items()):
            console.print(f"\n[bold cyan]{group_by.title()}: {group_name}[/bold cyan]")
            display_scores_table(group_scores, title=f"Quality Scores - {group_name}")
    else:
        display_scores_table(sorted_scores, title="Quality Score Comparison")

    # Output to file if requested
    if output:
        data = {"scores": [s.to_dict() for s in sorted_scores], "count": len(sorted_scores)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[green]Comparison results written to {output}[/green]")


# =============================================================================
# Summary Command
# =============================================================================


@app.command(name="summary")
@error_boundary
def summary_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file containing quality scores",
            exists=True,
        ),
    ],
) -> None:
    """Display summary statistics for quality scores.

    Shows aggregate statistics including counts by level, metric averages,
    and distribution of should_use recommendations.

    Examples:

        th quality summary scores.json
    """
    # Load scores
    scores = load_scores_from_file(input_file)

    if not scores:
        console.print("[yellow]No quality scores found in input file.[/yellow]")
        return

    # Calculate statistics
    from truthound.reporters.quality.base import QualityStatistics

    stats = QualityStatistics.from_scores(scores)

    # Display summary
    console.print(Panel("[bold]Quality Score Summary[/bold]", expand=False))
    console.print()

    # Total count
    console.print(f"[bold]Total Rules:[/bold] {stats.total_count}")
    console.print()

    # Quality level distribution
    console.print("[bold]Quality Level Distribution:[/bold]")
    level_table = Table(show_header=True, header_style="bold")
    level_table.add_column("Level")
    level_table.add_column("Count", justify="right")
    level_table.add_column("Percentage", justify="right")

    total = stats.total_count
    level_data = [
        ("Excellent", stats.excellent_count, "green"),
        ("Good", stats.good_count, "blue"),
        ("Acceptable", stats.acceptable_count, "yellow"),
        ("Poor", stats.poor_count, "dark_orange"),
        ("Unacceptable", stats.unacceptable_count, "red"),
    ]

    for level_name, count, color in level_data:
        pct = (count / total * 100) if total > 0 else 0
        level_table.add_row(
            f"[{color}]{level_name}[/{color}]",
            str(count),
            f"{pct:.1f}%",
        )

    console.print(level_table)
    console.print()

    # Recommendations
    console.print("[bold]Recommendations:[/bold]")
    console.print(f"  [green]Should Use:[/green] {stats.should_use_count}")
    console.print(f"  [red]Should Not Use:[/red] {stats.should_not_use_count}")
    console.print()

    # Metric averages
    console.print("[bold]Metric Averages:[/bold]")
    metrics_table = Table(show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Average", justify="right")
    metrics_table.add_column("Min", justify="right")
    metrics_table.add_column("Max", justify="right")

    metrics_data = [
        ("F1 Score", stats.avg_f1, stats.min_f1, stats.max_f1),
        ("Precision", stats.avg_precision, stats.min_precision, stats.max_precision),
        ("Recall", stats.avg_recall, stats.min_recall, stats.max_recall),
        ("Confidence", stats.avg_confidence, stats.min_confidence, stats.max_confidence),
    ]

    for name, avg, min_val, max_val in metrics_data:
        metrics_table.add_row(
            name,
            f"{avg:.2%}",
            f"{min_val:.2%}",
            f"{max_val:.2%}",
        )

    console.print(metrics_table)


# =============================================================================
# Module exports
# =============================================================================


def register_commands(parent_app: typer.Typer) -> None:
    """Register quality commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to.
    """
    parent_app.add_typer(app, name="quality")


__all__ = [
    "app",
    "register_commands",
    "report_cmd",
    "filter_cmd",
    "compare_cmd",
    "summary_cmd",
]
