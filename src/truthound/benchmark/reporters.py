"""Benchmark result reporters and exporters.

This module provides various output formats for benchmark results:
- ConsoleReporter: Rich terminal output
- JSONReporter: Machine-readable JSON
- MarkdownReporter: Documentation-friendly Markdown
- HTMLReporter: Interactive HTML reports

Each reporter can export individual results or suite results.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
import sys

from truthound.benchmark.base import (
    BenchmarkCategory,
    BenchmarkMetrics,
    BenchmarkResult,
    MetricUnit,
)
from truthound.benchmark.runner import SuiteResult


# =============================================================================
# Base Reporter
# =============================================================================


class BenchmarkReporter(ABC):
    """Abstract base class for benchmark result reporters.

    Defines the interface for exporting benchmark results in
    various formats.

    Example:
        reporter = MarkdownReporter()
        reporter.report_suite(suite_result, Path("benchmark_results.md"))
    """

    @abstractmethod
    def report_result(
        self,
        result: BenchmarkResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate report for a single benchmark result.

        Args:
            result: Benchmark result to report
            output: Output path or file-like object (optional)

        Returns:
            Report as string
        """
        pass

    @abstractmethod
    def report_suite(
        self,
        suite_result: SuiteResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate report for a suite of benchmark results.

        Args:
            suite_result: Suite results to report
            output: Output path or file-like object (optional)

        Returns:
            Report as string
        """
        pass

    def _write_output(
        self,
        content: str,
        output: Path | TextIO | None,
    ) -> None:
        """Write content to output destination."""
        if output is None:
            return

        if isinstance(output, Path):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content)
        else:
            output.write(content)


# =============================================================================
# Console Reporter
# =============================================================================


class ConsoleReporter(BenchmarkReporter):
    """Rich console output for benchmark results.

    Provides formatted terminal output with colors and alignment.

    Example:
        reporter = ConsoleReporter()
        print(reporter.report_suite(suite_result))
    """

    def __init__(
        self,
        use_colors: bool = True,
        show_details: bool = True,
    ):
        self.use_colors = use_colors
        self.show_details = show_details

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        if not self.use_colors:
            return text

        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.1f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.3f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"

    def _format_throughput(self, rows_per_sec: float) -> str:
        """Format throughput for display."""
        if rows_per_sec >= 1_000_000:
            return f"{rows_per_sec / 1_000_000:.2f}M rows/s"
        elif rows_per_sec >= 1_000:
            return f"{rows_per_sec / 1_000:.2f}K rows/s"
        else:
            return f"{rows_per_sec:.1f} rows/s"

    def report_result(
        self,
        result: BenchmarkResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate console report for a single result."""
        lines = []

        # Header
        status = self._color("✓ PASS", "green") if result.success else self._color("✗ FAIL", "red")
        lines.append(f"{self._color(result.benchmark_name, 'bold')} [{result.category.value}] {status}")
        lines.append("-" * 60)

        # Timing
        m = result.metrics
        lines.append(f"  Duration: {self._format_duration(m.mean_duration)} "
                     f"(±{self._format_duration(m.std_duration)})")
        lines.append(f"  Min/Max:  {self._format_duration(m.min_duration)} / "
                     f"{self._format_duration(m.max_duration)}")
        lines.append(f"  P95/P99:  {self._format_duration(m.p95_duration)} / "
                     f"{self._format_duration(m.p99_duration)}")

        # Throughput
        if m.rows_processed > 0:
            lines.append(f"  Throughput: {self._format_throughput(m.rows_per_second)}")

        # Memory
        if m.peak_memory_bytes > 0:
            peak_mb = m.peak_memory_bytes / (1024 * 1024)
            lines.append(f"  Peak Memory: {peak_mb:.1f} MB")

        # Error info
        if result.error:
            lines.append(f"  {self._color('Error:', 'red')} {result.error}")

        # Custom metrics
        if m.custom_metrics and self.show_details:
            lines.append("  Custom Metrics:")
            for metric in m.custom_metrics:
                lines.append(f"    {metric.name}: {metric.value:.2f} {metric.unit.value}")

        content = "\n".join(lines)
        self._write_output(content, output)
        return content

    def report_suite(
        self,
        suite_result: SuiteResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate console report for a suite."""
        lines = []

        # Header
        lines.append("")
        lines.append(self._color("=" * 70, "bold"))
        lines.append(self._color(f"  BENCHMARK SUITE: {suite_result.suite_name}", "bold"))
        lines.append(self._color("=" * 70, "bold"))
        lines.append("")

        # Environment
        env = suite_result.environment
        lines.append(f"Environment: Python {env.python_version} on {env.platform_system}")
        lines.append(f"Polars: {env.polars_version}, Truthound: {env.truthound_version}")
        lines.append(f"Timestamp: {suite_result.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        success_rate = suite_result.success_rate * 100
        color = "green" if success_rate == 100 else "yellow" if success_rate >= 80 else "red"
        lines.append(self._color(f"Results: {suite_result.successful_benchmarks}/{suite_result.total_benchmarks} passed ({success_rate:.0f}%)", color))
        lines.append(f"Total Duration: {self._format_duration(suite_result.total_duration_seconds)}")
        lines.append("")

        # Results by category
        by_category = suite_result.by_category
        for category, results in sorted(by_category.items(), key=lambda x: x[0].value):
            lines.append(self._color(f"  [{category.value.upper()}]", "blue"))

            for result in results:
                status = self._color("✓", "green") if result.success else self._color("✗", "red")
                duration = self._format_duration(result.metrics.mean_duration)

                throughput = ""
                if result.metrics.rows_processed > 0:
                    throughput = f" ({self._format_throughput(result.metrics.rows_per_second)})"

                lines.append(f"    {status} {result.benchmark_name}: {duration}{throughput}")

            lines.append("")

        # Failed benchmarks detail
        failed = suite_result.filter_failed()
        if failed:
            lines.append(self._color("FAILURES:", "red"))
            for result in failed:
                lines.append(f"  {result.benchmark_name}: {result.error}")
            lines.append("")

        lines.append("=" * 70)

        content = "\n".join(lines)
        self._write_output(content, output)
        return content


# =============================================================================
# JSON Reporter
# =============================================================================


class JSONReporter(BenchmarkReporter):
    """JSON format reporter for machine-readable output.

    Exports results in a structured JSON format suitable for
    CI/CD integration and programmatic analysis.

    Example:
        reporter = JSONReporter(pretty=True)
        reporter.report_suite(suite_result, Path("results.json"))
    """

    def __init__(
        self,
        pretty: bool = True,
        include_traceback: bool = False,
    ):
        self.pretty = pretty
        self.include_traceback = include_traceback

    def report_result(
        self,
        result: BenchmarkResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate JSON report for a single result."""
        data = result.to_dict()

        if not self.include_traceback:
            data.pop("error_traceback", None)

        indent = 2 if self.pretty else None
        content = json.dumps(data, indent=indent, default=str)

        self._write_output(content, output)
        return content

    def report_suite(
        self,
        suite_result: SuiteResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate JSON report for a suite."""
        data = suite_result.to_dict()

        if not self.include_traceback:
            for result in data.get("results", []):
                result.pop("error_traceback", None)

        indent = 2 if self.pretty else None
        content = json.dumps(data, indent=indent, default=str)

        self._write_output(content, output)
        return content


# =============================================================================
# Markdown Reporter
# =============================================================================


class MarkdownReporter(BenchmarkReporter):
    """Markdown format reporter for documentation.

    Generates Markdown tables and formatted text suitable for
    README files, documentation, and GitHub reports.

    Example:
        reporter = MarkdownReporter()
        reporter.report_suite(suite_result, Path("BENCHMARKS.md"))
    """

    def __init__(
        self,
        include_environment: bool = True,
        include_details: bool = True,
    ):
        self.include_environment = include_environment
        self.include_details = include_details

    def _format_duration(self, seconds: float) -> str:
        """Format duration for Markdown."""
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.1f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        else:
            return f"{seconds:.3f}s"

    def _format_throughput(self, rows_per_sec: float) -> str:
        """Format throughput for Markdown."""
        if rows_per_sec >= 1_000_000:
            return f"{rows_per_sec / 1_000_000:.2f}M rows/s"
        elif rows_per_sec >= 1_000:
            return f"{rows_per_sec / 1_000:.2f}K rows/s"
        else:
            return f"{rows_per_sec:.1f} rows/s"

    def report_result(
        self,
        result: BenchmarkResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate Markdown report for a single result."""
        lines = []

        status = "✅" if result.success else "❌"
        lines.append(f"## {result.benchmark_name} {status}")
        lines.append("")
        lines.append(f"**Category:** {result.category.value}")
        lines.append("")

        # Timing table
        m = result.metrics
        lines.append("### Timing")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean | {self._format_duration(m.mean_duration)} |")
        lines.append(f"| Std Dev | {self._format_duration(m.std_duration)} |")
        lines.append(f"| Min | {self._format_duration(m.min_duration)} |")
        lines.append(f"| Max | {self._format_duration(m.max_duration)} |")
        lines.append(f"| P95 | {self._format_duration(m.p95_duration)} |")
        lines.append(f"| P99 | {self._format_duration(m.p99_duration)} |")
        lines.append("")

        # Throughput
        if m.rows_processed > 0:
            lines.append("### Throughput")
            lines.append("")
            lines.append(f"- **Rows processed:** {m.rows_processed:,}")
            lines.append(f"- **Throughput:** {self._format_throughput(m.rows_per_second)}")
            if m.mb_per_second > 0:
                lines.append(f"- **Data rate:** {m.mb_per_second:.2f} MB/s")
            lines.append("")

        # Error
        if result.error:
            lines.append("### Error")
            lines.append("")
            lines.append(f"```\n{result.error}\n```")
            lines.append("")

        content = "\n".join(lines)
        self._write_output(content, output)
        return content

    def report_suite(
        self,
        suite_result: SuiteResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate Markdown report for a suite."""
        lines = []

        # Title
        lines.append(f"# Benchmark Results: {suite_result.suite_name}")
        lines.append("")
        lines.append(f"*Generated: {suite_result.started_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        # Environment
        if self.include_environment:
            env = suite_result.environment
            lines.append("## Environment")
            lines.append("")
            lines.append(f"- **Python:** {env.python_version}")
            lines.append(f"- **Platform:** {env.platform_system} {env.platform_release} ({env.platform_machine})")
            lines.append(f"- **CPU Cores:** {env.cpu_count}")
            lines.append(f"- **Polars:** {env.polars_version}")
            lines.append(f"- **Truthound:** {env.truthound_version}")
            lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        success_rate = suite_result.success_rate * 100
        emoji = "🟢" if success_rate == 100 else "🟡" if success_rate >= 80 else "🔴"
        lines.append(f"| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Benchmarks | {suite_result.total_benchmarks} |")
        lines.append(f"| Successful | {suite_result.successful_benchmarks} |")
        lines.append(f"| Failed | {suite_result.failed_benchmarks} |")
        lines.append(f"| Success Rate | {emoji} {success_rate:.0f}% |")
        lines.append(f"| Total Duration | {suite_result.total_duration_seconds:.2f}s |")
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Benchmark | Category | Status | Duration | Throughput |")
        lines.append("|-----------|----------|--------|----------|------------|")

        for result in suite_result.results:
            status = "✅" if result.success else "❌"
            duration = self._format_duration(result.metrics.mean_duration)

            throughput = "-"
            if result.metrics.rows_processed > 0:
                throughput = self._format_throughput(result.metrics.rows_per_second)

            lines.append(f"| {result.benchmark_name} | {result.category.value} | {status} | {duration} | {throughput} |")

        lines.append("")

        # Detailed results
        if self.include_details:
            lines.append("## Detailed Results")
            lines.append("")

            for result in suite_result.results:
                lines.append(self.report_result(result))

        # Failed benchmarks
        failed = suite_result.filter_failed()
        if failed:
            lines.append("## Failures")
            lines.append("")
            for result in failed:
                lines.append(f"### {result.benchmark_name}")
                lines.append("")
                lines.append(f"```\n{result.error}\n```")
                lines.append("")

        content = "\n".join(lines)
        self._write_output(content, output)
        return content


# =============================================================================
# HTML Reporter
# =============================================================================

# Check for jinja2 availability
try:
    from jinja2 import Environment, BaseLoader, select_autoescape

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


def _require_jinja2() -> None:
    """Check if jinja2 is available."""
    if not HAS_JINJA2:
        raise ImportError(
            "jinja2 is required for HTML reports. "
            "Install with: pip install truthound[reports]"
        )


# HTML template for benchmark reports
_BENCHMARK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .summary-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card .value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .summary-card .label { color: #7f8c8d; }
        .success { color: #27ae60; }
        .failure { color: #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .status-pass { color: #27ae60; font-weight: bold; }
        .status-fail { color: #e74c3c; font-weight: bold; }
        .environment { background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .environment code { background: #bdc3c7; padding: 2px 6px; border-radius: 3px; }
        .metric-bar { background: #ecf0f1; border-radius: 4px; overflow: hidden; height: 20px; }
        .metric-fill { background: #3498db; height: 100%; }
        footer { text-align: center; margin-top: 30px; color: #7f8c8d; }
        footer a { color: #3498db; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>Suite: <strong>{{ suite_name }}</strong></p>
        <p>Generated: {{ generated_at }}</p>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{{ total_benchmarks }}</div>
                <div class="label">Total Benchmarks</div>
            </div>
            <div class="summary-card">
                <div class="value success">{{ successful_benchmarks }}</div>
                <div class="label">Successful</div>
            </div>
            <div class="summary-card">
                <div class="value failure">{{ failed_benchmarks }}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ success_rate }}%</div>
                <div class="label">Success Rate</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ total_duration }}</div>
                <div class="label">Total Duration</div>
            </div>
        </div>

        <div class="environment">
            <strong>Environment:</strong>
            Python <code>{{ env.python_version }}</code> on
            <code>{{ env.platform_system }} {{ env.platform_release }}</code>
            ({{ env.cpu_count }} CPUs) |
            Polars <code>{{ env.polars_version }}</code> |
            Truthound <code>{{ env.truthound_version }}</code>
        </div>

        <h2>Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Throughput</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td>{{ result.name }}</td>
                    <td>{{ result.category }}</td>
                    <td class="{{ result.status_class }}">{{ result.status_text }}</td>
                    <td>{{ result.duration }}</td>
                    <td>{{ result.throughput }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <footer>
            <p>Generated by <a href="https://github.com/seadonggyun4/Truthound">Truthound</a> Benchmark System</p>
        </footer>
    </div>
</body>
</html>"""

# Single result template
_BENCHMARK_RESULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ benchmark_name }} - Benchmark Result</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        .status-pass { color: #27ae60; font-weight: bold; }
        .status-fail { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ benchmark_name }}</h1>
        <p><strong>Category:</strong> {{ category }}</p>
        <p><strong>Status:</strong> <span class="{{ status_class }}">{{ status_text }}</span></p>

        <h2>Timing Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean Duration</td><td>{{ mean_duration }}</td></tr>
            <tr><td>Std Deviation</td><td>{{ std_duration }}</td></tr>
            <tr><td>Min</td><td>{{ min_duration }}</td></tr>
            <tr><td>Max</td><td>{{ max_duration }}</td></tr>
            <tr><td>P95</td><td>{{ p95_duration }}</td></tr>
            <tr><td>P99</td><td>{{ p99_duration }}</td></tr>
        </table>
    </div>
</body>
</html>"""


class HTMLReporter(BenchmarkReporter):
    """HTML format reporter for interactive reports.

    Generates styled HTML reports using Jinja2 templates.
    Requires jinja2: pip install truthound[reports]

    Example:
        reporter = HTMLReporter()
        reporter.report_suite(suite_result, Path("benchmark_report.html"))
    """

    def __init__(
        self,
        title: str = "Truthound Benchmark Report",
        include_charts: bool = True,
    ):
        _require_jinja2()
        self.title = title
        self.include_charts = include_charts
        self._env = self._create_environment()

    def _create_environment(self) -> "Environment":
        """Create Jinja2 environment with custom filters."""
        _require_jinja2()

        class StringLoader(BaseLoader):
            def get_source(self, environment, template):
                templates = {
                    "suite": _BENCHMARK_HTML_TEMPLATE,
                    "result": _BENCHMARK_RESULT_TEMPLATE,
                }
                if template in templates:
                    return templates[template], None, lambda: False
                raise ValueError(f"Unknown template: {template}")

        env = Environment(
            loader=StringLoader(),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Add custom filters
        env.filters["format_duration"] = self._format_duration
        env.filters["format_throughput"] = self._format_throughput

        return env

    def _format_duration(self, seconds: float) -> str:
        """Format duration for HTML."""
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.1f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        else:
            return f"{seconds:.3f}s"

    def _format_throughput(self, rows_per_sec: float) -> str:
        """Format throughput for HTML."""
        if rows_per_sec >= 1_000_000:
            return f"{rows_per_sec / 1_000_000:.2f}M rows/s"
        elif rows_per_sec >= 1_000:
            return f"{rows_per_sec / 1_000:.2f}K rows/s"
        else:
            return f"{rows_per_sec:.1f} rows/s"

    def report_result(
        self,
        result: BenchmarkResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate HTML report for a single result."""
        template = self._env.get_template("result")

        m = result.metrics
        html = template.render(
            benchmark_name=result.benchmark_name,
            category=result.category.value,
            status_class="status-pass" if result.success else "status-fail",
            status_text="PASS" if result.success else "FAIL",
            mean_duration=self._format_duration(m.mean_duration),
            std_duration=self._format_duration(m.std_duration),
            min_duration=self._format_duration(m.min_duration),
            max_duration=self._format_duration(m.max_duration),
            p95_duration=self._format_duration(m.p95_duration),
            p99_duration=self._format_duration(m.p99_duration),
        )

        self._write_output(html, output)
        return html

    def report_suite(
        self,
        suite_result: SuiteResult,
        output: Path | TextIO | None = None,
    ) -> str:
        """Generate HTML report for a suite."""
        template = self._env.get_template("suite")

        # Prepare results data
        results_data = []
        for result in suite_result.results:
            throughput = "-"
            if result.metrics.rows_processed > 0:
                throughput = self._format_throughput(result.metrics.rows_per_second)

            results_data.append({
                "name": result.benchmark_name,
                "category": result.category.value,
                "status_class": "status-pass" if result.success else "status-fail",
                "status_text": "✓ PASS" if result.success else "✗ FAIL",
                "duration": self._format_duration(result.metrics.mean_duration),
                "throughput": throughput,
            })

        html = template.render(
            title=self.title,
            suite_name=suite_result.suite_name,
            generated_at=suite_result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            total_benchmarks=suite_result.total_benchmarks,
            successful_benchmarks=suite_result.successful_benchmarks,
            failed_benchmarks=suite_result.failed_benchmarks,
            success_rate=f"{suite_result.success_rate * 100:.0f}",
            total_duration=f"{suite_result.total_duration_seconds:.2f}s",
            env=suite_result.environment,
            results=results_data,
        )

        self._write_output(html, output)
        return html
