"""Benchmark commands.

This module implements performance benchmarking commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary

# Known benchmark names for better error messages
_KNOWN_BENCHMARKS = {"profile", "check", "scan", "compare", "learn", "throughput"}


# Benchmark app for subcommands
app = typer.Typer(
    name="benchmark",
    help="""Performance benchmarking commands.

Subcommands: run, list, compare

Quick start:
  truthound benchmark run --suite quick      # Run quick benchmark suite
  truthound benchmark run profile            # Run single 'profile' benchmark
  truthound benchmark list                   # List available benchmarks
  truthound benchmark compare a.json b.json  # Compare results
""",
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
            "--suite", "-s", help="Predefined suite: quick (~5s), ci (~15s), full (~30s)"
        ),
    ] = None,
    size: Annotated[
        str,
        typer.Option("--size", help="Data size: tiny (1K), small (10K), medium (100K), large (1M)"),
    ] = "small",
    rows: Annotated[
        Optional[int],
        typer.Option("--rows", "-r", help="Custom row count (overrides size)"),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Number of measurement iterations"),
    ] = 3,
    warmup: Annotated[
        int,
        typer.Option("--warmup", "-w", help="Number of warmup iterations"),
    ] = 1,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        Optional[str],
        typer.Option("--format", "-f", help="Output format (json, html). Auto-detected from -o extension if not specified"),
    ] = None,
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

    Estimated times (default settings):
        --suite quick:  ~5 seconds  (1K rows, 3 benchmarks)
        --suite ci:     ~15 seconds (10K rows, 5 benchmarks)
        --suite full:   ~30 seconds (10K rows, 6 benchmarks)

    Examples:
        truthound benchmark run --suite quick     # Fast feedback
        truthound benchmark run --suite ci        # CI/CD appropriate
        truthound benchmark run profile           # Single benchmark
        truthound benchmark run check --rows 100000 --iterations 5
    """
    from truthound.benchmark import (
        BenchmarkRunner,
        BenchmarkSuite,
        BenchmarkConfig,
        BenchmarkSize,
        RunnerConfig,
        ConsoleReporter,
        JSONReporter,
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

        # Determine output format
        output_format = format
        if output and not output_format:
            # Auto-detect from file extension
            ext = output.suffix.lower()
            if ext == ".json":
                output_format = "json"
            elif ext in (".html", ".htm"):
                output_format = "html"
            else:
                # Default to JSON for unknown extensions when saving to file
                output_format = "json"
        elif not output_format:
            # No output file, default to console
            output_format = "console"

        # Validate format
        if output_format not in ("console", "json", "html"):
            typer.echo(f"Unknown format: {output_format}. Use json or html.", err=True)
            raise typer.Exit(1)

        # Generate output
        reporters = {
            "console": ConsoleReporter(use_colors=True),
            "json": JSONReporter(pretty=True),
            "html": HTMLReporter(),
        }

        reporter = reporters.get(output_format, JSONReporter(pretty=True))
        report_content = reporter.report_suite(results)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report_content)
            typer.echo(f"Results saved to: {output} (format: {output_format})")
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
        typer.Option("--format", "-f", help="Output format (console, json)"),
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

    # Check file extensions and warn if not JSON
    for file_path, name in [(baseline, "baseline"), (current, "current")]:
        if file_path.suffix.lower() not in (".json",):
            typer.echo(
                f"Warning: {name} file '{file_path}' does not have .json extension.\n"
                f"This command compares benchmark result JSON files, not data files.\n\n"
                f"Did you mean to run a benchmark first?\n"
                f"  truthound benchmark run --suite ci -o {name}.json\n",
                err=True,
            )

    try:
        baseline_data = json.loads(baseline.read_text())
        current_data = json.loads(current.read_text())

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

        # Output based on format
        if format == "json":
            output_data = {
                "baseline_file": str(baseline),
                "current_file": str(current),
                "threshold_percent": threshold,
                "regressions": [
                    {
                        "name": name,
                        "baseline_seconds": base_d,
                        "current_seconds": curr_d,
                        "change_percent": pct,
                    }
                    for name, base_d, curr_d, pct in regressions
                ],
                "improvements": [
                    {
                        "name": name,
                        "baseline_seconds": base_d,
                        "current_seconds": curr_d,
                        "change_percent": pct,
                    }
                    for name, base_d, curr_d, pct in improvements
                ],
                "has_regressions": len(regressions) > 0,
            }
            typer.echo(json.dumps(output_data, indent=2))
        else:
            # Console format
            typer.echo("\nBenchmark Comparison")
            typer.echo("=" * 60)
            typer.echo(f"Baseline: {baseline}")
            typer.echo(f"Current:  {current}")
            typer.echo(f"Threshold: {threshold}%")
            typer.echo("-" * 60)

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
    except json.JSONDecodeError as e:
        typer.echo(
            f"Error: Failed to parse JSON - {e}\n\n"
            f"Both baseline and current must be JSON files from 'benchmark run'.\n"
            f"Example workflow:\n"
            f"  1. truthound benchmark run --suite ci -o baseline.json\n"
            f"  2. (make changes)\n"
            f"  3. truthound benchmark run --suite ci -o current.json\n"
            f"  4. truthound benchmark compare baseline.json current.json",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
