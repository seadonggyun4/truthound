"""Report generation for stress tests.

Provides comprehensive reporting for stress test results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import logging

from tests.stress.framework import StressTestResult
from tests.stress.metrics import StressMetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class StressTestReport:
    """Comprehensive stress test report.

    Aggregates results from a stress test run with
    detailed analysis and recommendations.
    """

    test_name: str
    result: StressTestResult
    metrics: StressMetricsCollector | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Generate recommendations based on results."""
        self._analyze_results()

    def _analyze_results(self) -> None:
        """Analyze results and generate recommendations."""
        result = self.result

        # Check success rate
        if result.success_rate < 0.99:
            if result.success_rate < 0.95:
                self.warnings.append(
                    f"Critical: Success rate ({result.success_rate:.2%}) is below 95%"
                )
                self.recommendations.append(
                    "Investigate root cause of failures immediately"
                )
            else:
                self.warnings.append(
                    f"Warning: Success rate ({result.success_rate:.2%}) is below 99%"
                )
                self.recommendations.append(
                    "Review error logs and consider adding retry logic"
                )

        # Check P99 latency
        if result.latency_p99 > 1000:
            if result.latency_p99 > 5000:
                self.warnings.append(
                    f"Critical: P99 latency ({result.latency_p99:.0f}ms) exceeds 5 seconds"
                )
                self.recommendations.append(
                    "Consider adding caching or optimizing slow queries"
                )
            else:
                self.warnings.append(
                    f"Warning: P99 latency ({result.latency_p99:.0f}ms) exceeds 1 second"
                )
                self.recommendations.append(
                    "Profile slow operations and optimize hot paths"
                )

        # Check for latency variance
        if result.latencies_ms:
            avg = result.latency_avg
            p99 = result.latency_p99
            if p99 > avg * 10:
                self.warnings.append(
                    f"High latency variance: P99 ({p99:.0f}ms) is 10x+ avg ({avg:.0f}ms)"
                )
                self.recommendations.append(
                    "Investigate outlier requests causing tail latency"
                )

        # Check throughput stability
        phases = result.phases
        if phases:
            steady_ops = phases.get("steady", {}).get("operations", 0)
            steady_failures = phases.get("steady", {}).get("failures", 0)
            if steady_ops > 0:
                steady_success_rate = 1 - (steady_failures / steady_ops)
                if steady_success_rate < result.success_rate:
                    self.warnings.append(
                        "Performance degradation under sustained load"
                    )
                    self.recommendations.append(
                        "Check for resource leaks or connection pool exhaustion"
                    )

        # General recommendations if test passed
        if result.passed and not self.warnings:
            self.recommendations.append(
                "System is performing within acceptable parameters"
            )
            self.recommendations.append(
                "Consider increasing load to find breaking point"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        report_dict = {
            "test_name": self.test_name,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.result.passed,
            "summary": {
                "duration_seconds": self.result.duration_seconds,
                "total_operations": self.result.total_operations,
                "success_rate": self.result.success_rate,
                "throughput_per_second": self.result.throughput_per_second,
            },
            "latency": {
                "avg_ms": self.result.latency_avg,
                "p50_ms": self.result.latency_p50,
                "p95_ms": self.result.latency_p95,
                "p99_ms": self.result.latency_p99,
            },
            "phases": self.result.phases,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "errors": self.result.errors[:20],  # Limit errors
        }

        if self.metrics:
            report_dict["detailed_metrics"] = self.metrics.get_summary()

        return report_dict

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = [
            f"# Stress Test Report: {self.test_name}",
            "",
            f"**Generated:** {self.timestamp.isoformat()}",
            f"**Status:** {'PASSED' if self.result.passed else 'FAILED'}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Duration | {self.result.duration_seconds:.1f}s |",
            f"| Total Operations | {self.result.total_operations:,} |",
            f"| Successful | {self.result.successful_operations:,} |",
            f"| Failed | {self.result.failed_operations:,} |",
            f"| Success Rate | {self.result.success_rate:.2%} |",
            f"| Throughput | {self.result.throughput_per_second:.1f} ops/sec |",
            "",
            "## Latency",
            "",
            f"| Percentile | Value |",
            f"|------------|-------|",
            f"| Average | {self.result.latency_avg:.1f}ms |",
            f"| P50 | {self.result.latency_p50:.1f}ms |",
            f"| P95 | {self.result.latency_p95:.1f}ms |",
            f"| P99 | {self.result.latency_p99:.1f}ms |",
            "",
        ]

        # Phase breakdown
        if self.result.phases:
            lines.extend([
                "## Phase Breakdown",
                "",
                "| Phase | Operations | Successes | Failures |",
                "|-------|------------|-----------|----------|",
            ])
            for phase_name, stats in self.result.phases.items():
                lines.append(
                    f"| {phase_name} | {stats.get('operations', 0):,} | "
                    f"{stats.get('successes', 0):,} | {stats.get('failures', 0):,} |"
                )
            lines.append("")

        # Warnings
        if self.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Errors
        if self.result.errors:
            lines.extend([
                "## Sample Errors",
                "",
                "```",
            ])
            for error in self.result.errors[:10]:
                lines.append(error)
            lines.extend([
                "```",
                "",
            ])

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert report to HTML format."""
        status_color = "#28a745" if self.result.passed else "#dc3545"
        status_text = "PASSED" if self.result.passed else "FAILED"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stress Test Report: {self.test_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; background-color: {status_color}; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .warning {{ color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .recommendation {{ color: #004085; background-color: #cce5ff; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .error {{ font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Stress Test Report: {self.test_name}</h1>
    <p><strong>Generated:</strong> {self.timestamp.isoformat()}</p>
    <p><strong>Status:</strong> <span class="status">{status_text}</span></p>

    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Duration</td><td>{self.result.duration_seconds:.1f}s</td></tr>
        <tr><td>Total Operations</td><td>{self.result.total_operations:,}</td></tr>
        <tr><td>Successful</td><td>{self.result.successful_operations:,}</td></tr>
        <tr><td>Failed</td><td>{self.result.failed_operations:,}</td></tr>
        <tr><td>Success Rate</td><td>{self.result.success_rate:.2%}</td></tr>
        <tr><td>Throughput</td><td>{self.result.throughput_per_second:.1f} ops/sec</td></tr>
    </table>

    <h2>Latency</h2>
    <table>
        <tr><th>Percentile</th><th>Value</th></tr>
        <tr><td>Average</td><td>{self.result.latency_avg:.1f}ms</td></tr>
        <tr><td>P50</td><td>{self.result.latency_p50:.1f}ms</td></tr>
        <tr><td>P95</td><td>{self.result.latency_p95:.1f}ms</td></tr>
        <tr><td>P99</td><td>{self.result.latency_p99:.1f}ms</td></tr>
    </table>
"""

        if self.warnings:
            html += "\n    <h2>Warnings</h2>\n"
            for warning in self.warnings:
                html += f'    <div class="warning">{warning}</div>\n'

        if self.recommendations:
            html += "\n    <h2>Recommendations</h2>\n"
            for rec in self.recommendations:
                html += f'    <div class="recommendation">{rec}</div>\n'

        if self.result.errors:
            html += "\n    <h2>Sample Errors</h2>\n"
            html += '    <div class="error">\n'
            for error in self.result.errors[:10]:
                html += f"        {error}<br>\n"
            html += "    </div>\n"

        html += """
</body>
</html>
"""
        return html


class ReportGenerator:
    """Generator for stress test reports.

    Handles report generation in multiple formats
    and saving to files.

    Example:
        >>> generator = ReportGenerator(output_dir="./reports")
        >>> generator.generate(result, format="all")
    """

    def __init__(
        self,
        output_dir: str | Path = "./stress_reports",
    ) -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: StressTestResult,
        metrics: StressMetricsCollector | None = None,
        format: str = "all",
    ) -> list[Path]:
        """Generate reports in specified format(s).

        Args:
            result: Stress test result.
            metrics: Optional detailed metrics.
            format: "json", "markdown", "html", or "all".

        Returns:
            List of generated report file paths.
        """
        report = StressTestReport(
            test_name=result.config.name,
            result=result,
            metrics=metrics,
        )

        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.config.name}_{timestamp}"

        formats_to_generate = (
            ["json", "markdown", "html"] if format == "all" else [format]
        )

        for fmt in formats_to_generate:
            try:
                if fmt == "json":
                    file_path = self._output_dir / f"{base_name}.json"
                    file_path.write_text(report.to_json())
                    generated_files.append(file_path)

                elif fmt == "markdown":
                    file_path = self._output_dir / f"{base_name}.md"
                    file_path.write_text(report.to_markdown())
                    generated_files.append(file_path)

                elif fmt == "html":
                    file_path = self._output_dir / f"{base_name}.html"
                    file_path.write_text(report.to_html())
                    generated_files.append(file_path)

                logger.info(f"Generated report: {file_path}")

            except Exception as e:
                logger.error(f"Failed to generate {fmt} report: {e}")

        return generated_files

    def generate_comparison(
        self,
        results: list[StressTestResult],
        output_name: str = "comparison",
    ) -> Path:
        """Generate a comparison report for multiple test runs.

        Args:
            results: List of stress test results.
            output_name: Output file name (without extension).

        Returns:
            Path to generated comparison report.
        """
        comparison = {
            "generated": datetime.now().isoformat(),
            "test_count": len(results),
            "tests": [],
        }

        for result in results:
            comparison["tests"].append({
                "name": result.config.name,
                "passed": result.passed,
                "success_rate": result.success_rate,
                "throughput": result.throughput_per_second,
                "latency_p99": result.latency_p99,
                "duration": result.duration_seconds,
            })

        # Sort by success rate, then throughput
        comparison["tests"].sort(
            key=lambda x: (x["success_rate"], x["throughput"]),
            reverse=True,
        )

        # Generate markdown comparison
        lines = [
            "# Stress Test Comparison",
            "",
            f"**Generated:** {comparison['generated']}",
            f"**Tests Compared:** {comparison['test_count']}",
            "",
            "## Results",
            "",
            "| Test | Status | Success Rate | Throughput | P99 Latency | Duration |",
            "|------|--------|--------------|------------|-------------|----------|",
        ]

        for test in comparison["tests"]:
            status = "" if test["passed"] else ""
            lines.append(
                f"| {test['name']} | {status} | "
                f"{test['success_rate']:.2%} | "
                f"{test['throughput']:.1f}/s | "
                f"{test['latency_p99']:.0f}ms | "
                f"{test['duration']:.1f}s |"
            )

        file_path = self._output_dir / f"{output_name}.md"
        file_path.write_text("\n".join(lines))

        logger.info(f"Generated comparison report: {file_path}")
        return file_path
