"""Benchmark commands.

This module implements performance benchmarking commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary

# Benchmark app for subcommands
app = typer.Typer(
    name="benchmark",
    help="Performance benchmarking commands",
)


@app.command(name="run")
@error_boundary
def run_cmd(
    benchmark: Annotated[
        Optional[str],
        typer.Argument(help="Benchmark name to run (e.g., 'profile', 'check')"),
    ] = None,
    suite: Annotated[
        Optional[str],
        typer.Option(
            "--suite", "-s", help="Predefined suite to run (quick, ci, full, profiling, validation)"
        ),
    ] = None,
    size: Annotated[
        str,
        typer.Option("--size", help="Data size (tiny, small, medium, large, xlarge)"),
    ] = "medium",
    rows: Annotated[
        Optional[int],
        typer.Option("--rows", "-r", help="Custom row count (overrides size)"),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Number of measurement iterations"),
    ] = 5,
    warmup: Annotated[
        int,
        typer.Option("--warmup", "-w", help="Number of warmup iterations"),
    ] = 2,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, markdown, html)"),
    ] = "console",
    save_baseline: Annotated[
        bool,
        typer.Option("--save-baseline", help="Save results as baseline for regression detection"),
    ] = False,
    compare_baseline: Annotated[
        bool,
        typer.Option("--compare-baseline", help="Compare against saved baseline"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Run performance benchmarks.

    Examples:
        truthound benchmark run profile --size medium
        truthound benchmark run --suite quick
        truthound benchmark run check --rows 1000000
        truthound benchmark run --suite ci --save-baseline
        truthound benchmark run --suite ci --compare-baseline
    """
    from truthound.benchmark import (
        BenchmarkRunner,
        BenchmarkSuite,
        BenchmarkConfig,
        BenchmarkSize,
        RunnerConfig,
        ConsoleReporter,
        JSONReporter,
        MarkdownReporter,
        HTMLReporter,
        RegressionDetector,
    )

    try:
        # Determine row count
        size_map = {
            "tiny": BenchmarkSize.TINY,
            "small": BenchmarkSize.SMALL,
            "medium": BenchmarkSize.MEDIUM,
            "large": BenchmarkSize.LARGE,
            "xlarge": BenchmarkSize.XLARGE,
        }
        benchmark_size = size_map.get(size, BenchmarkSize.MEDIUM)
        row_count = rows if rows else benchmark_size.row_count

        # Configure benchmark
        benchmark_config = BenchmarkConfig(
            warmup_iterations=warmup,
            measure_iterations=iterations,
            default_size=benchmark_size,
            verbose=verbose,
        )

        runner_config = RunnerConfig(
            size_override=benchmark_size if not rows else None,
            verbose=verbose,
        )

        runner = BenchmarkRunner(
            config=runner_config,
            benchmark_config=benchmark_config,
        )

        # Determine what to run
        if suite:
            suite_map = {
                "quick": BenchmarkSuite.quick,
                "ci": BenchmarkSuite.ci,
                "full": lambda: BenchmarkSuite.full(benchmark_size),
                "profiling": lambda: BenchmarkSuite.profiling(benchmark_size),
                "validation": lambda: BenchmarkSuite.validation(benchmark_size),
            }
            if suite not in suite_map:
                typer.echo(
                    f"Unknown suite: {suite}. Available: {list(suite_map.keys())}",
                    err=True,
                )
                raise typer.Exit(1)

            benchmark_suite = suite_map[suite]()
            results = runner.run_suite(benchmark_suite)

        elif benchmark:
            result = runner.run(benchmark, row_count=row_count)
            # Wrap single result in suite result
            from truthound.benchmark.base import EnvironmentInfo
            from truthound.benchmark.runner import SuiteResult

            results = SuiteResult(
                suite_name=f"single:{benchmark}",
                results=[result],
                environment=EnvironmentInfo.capture(),
            )
            results.completed_at = result.completed_at

        else:
            typer.echo("Specify either a benchmark name or --suite", err=True)
            raise typer.Exit(1)

        # Compare against baseline if requested
        if compare_baseline:
            detector = RegressionDetector()
            report = detector.generate_report(results)
            typer.echo(report)

            regressions = detector.check(results)
            if regressions:
                typer.echo("\nPerformance regressions detected!", err=True)
                raise typer.Exit(1)

        # Generate output
        reporters = {
            "console": ConsoleReporter(use_colors=True),
            "json": JSONReporter(pretty=True),
            "markdown": MarkdownReporter(),
            "html": HTMLReporter(),
        }

        reporter = reporters.get(format, ConsoleReporter())
        report_content = reporter.report_suite(results)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report_content)
            typer.echo(f"Results saved to: {output}")
        elif format == "console":
            typer.echo(report_content)
        else:
            typer.echo(report_content)

        # Save baseline if requested
        if save_baseline:
            detector = RegressionDetector()
            detector.save_baseline(results)
            typer.echo(f"Baseline saved to: {detector.history_path}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command(name="list")
@error_boundary
def list_cmd(
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """List available benchmarks.

    Displays all registered benchmarks grouped by category.
    """
    from truthound.benchmark import benchmark_registry

    benchmarks = benchmark_registry.list_all()

    if format == "json":
        data = [
            {
                "name": b.name,
                "category": b.category.value,
                "description": b.description,
            }
            for b in benchmarks
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo("\nAvailable Benchmarks:")
        typer.echo("=" * 60)

        # Group by category
        from collections import defaultdict

        by_category = defaultdict(list)
        for b in benchmarks:
            by_category[b.category.value].append(b)

        for category in sorted(by_category.keys()):
            typer.echo(f"\n[{category.upper()}]")
            for b in by_category[category]:
                typer.echo(f"  {b.name:20} - {b.description}")


@app.command(name="compare")
@error_boundary
def compare_cmd(
    baseline: Annotated[
        Path,
        typer.Argument(help="Baseline results JSON file"),
    ],
    current: Annotated[
        Path,
        typer.Argument(help="Current results JSON file"),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Regression threshold percentage"),
    ] = 10.0,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, markdown)"),
    ] = "console",
) -> None:
    """Compare two benchmark results for regressions.

    This command compares benchmark results and reports any
    performance regressions or improvements.

    Examples:
        truthound benchmark compare baseline.json current.json
        truthound benchmark compare old.json new.json --threshold 5.0
    """
    from truthound.cli_modules.common.errors import require_file

    require_file(baseline, "Baseline file")
    require_file(current, "Current file")

    try:
        baseline_data = json.loads(baseline.read_text())
        current_data = json.loads(current.read_text())

        typer.echo("\nBenchmark Comparison")
        typer.echo("=" * 60)
        typer.echo(f"Baseline: {baseline}")
        typer.echo(f"Current:  {current}")
        typer.echo(f"Threshold: {threshold}%")
        typer.echo("-" * 60)

        baseline_results = {
            r["benchmark_name"]: r for r in baseline_data.get("results", [])
        }
        current_results = {r["benchmark_name"]: r for r in current_data.get("results", [])}

        regressions = []
        improvements = []

        for name, curr in current_results.items():
            if name not in baseline_results:
                continue

            base = baseline_results[name]
            base_duration = base["metrics"]["timing"]["mean_seconds"]
            curr_duration = curr["metrics"]["timing"]["mean_seconds"]

            if base_duration > 0:
                pct_change = ((curr_duration - base_duration) / base_duration) * 100

                if pct_change > threshold:
                    regressions.append((name, base_duration, curr_duration, pct_change))
                elif pct_change < -threshold:
                    improvements.append((name, base_duration, curr_duration, pct_change))

        if regressions:
            typer.echo("\nREGRESSIONS:")
            for name, base_d, curr_d, pct in regressions:
                typer.echo(f"  {name}: {base_d:.3f}s -> {curr_d:.3f}s ({pct:+.1f}%)")

        if improvements:
            typer.echo("\nIMPROVEMENTS:")
            for name, base_d, curr_d, pct in improvements:
                typer.echo(f"  {name}: {base_d:.3f}s -> {curr_d:.3f}s ({pct:+.1f}%)")

        if not regressions and not improvements:
            typer.echo("\nNo significant changes detected.")

        typer.echo("")

        if regressions:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
