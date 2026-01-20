"""HTML report generation bridge module.

This module provides a bridge between the CLI's Report objects and the
HTMLReporter system. It converts Report objects to ValidationResult
objects and generates HTML reports.

Example:
    >>> from truthound.html_reporter import generate_html_report
    >>> from truthound.report import Report
    >>> report = Report(issues=[], source="data.csv")
    >>> html = generate_html_report(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from truthound.report import Report, PIIReport

# Check for jinja2 availability
try:
    import jinja2  # noqa: F401

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


def _require_jinja2() -> None:
    """Check if jinja2 is available and raise helpful error if not."""
    if not HAS_JINJA2:
        raise ImportError(
            "HTML reports require jinja2. "
            "Install with: pip install truthound[reports] or pip install jinja2"
        )


@dataclass
class HTMLReportConfig:
    """Configuration for HTML report generation.

    Attributes:
        title: Report title displayed in the header.
        theme: Color theme ("light" or "dark").
        custom_css: Additional CSS to inject into the report.
        include_metadata: Whether to include metadata section.
        include_statistics: Whether to include statistics section.
    """

    title: str = "Truthound Validation Report"
    theme: str = "light"
    custom_css: str = ""
    include_metadata: bool = True
    include_statistics: bool = True


def generate_html_report(
    report: "Report",
    config: HTMLReportConfig | None = None,
    **kwargs: Any,
) -> str:
    """Generate an HTML report from a Report object.

    This function converts the Report object to a ValidationResult and
    uses the HTMLReporter to generate a styled HTML report.

    Args:
        report: The Report object containing validation issues.
        config: Optional configuration for the HTML report.
        **kwargs: Additional keyword arguments passed to HTMLReporter.

    Returns:
        HTML string representation of the report.

    Raises:
        ImportError: If jinja2 is not installed.

    Example:
        >>> from truthound.api import check
        >>> from truthound.html_reporter import generate_html_report
        >>> report = check("data.csv")
        >>> html = generate_html_report(report)
        >>> with open("report.html", "w") as f:
        ...     f.write(html)
    """
    _require_jinja2()

    from truthound.reporters.html_reporter import HTMLReporter
    from truthound.stores.results import ValidationResult

    # Apply config if provided
    if config:
        kwargs.setdefault("title", config.title)
        kwargs.setdefault("theme", config.theme)
        kwargs.setdefault("custom_css", config.custom_css)
        kwargs.setdefault("include_metadata", config.include_metadata)
        kwargs.setdefault("include_statistics", config.include_statistics)

    # Convert Report to ValidationResult
    validation_result = ValidationResult.from_report(
        report=report,
        data_asset=report.source,
    )

    # Create reporter and render
    reporter = HTMLReporter(**kwargs)
    return reporter.render(validation_result)


def write_html_report(
    report: "Report",
    output_path: str | Path,
    config: HTMLReportConfig | None = None,
    **kwargs: Any,
) -> Path:
    """Generate and write an HTML report to a file.

    Args:
        report: The Report object containing validation issues.
        output_path: Path where the HTML file should be written.
        config: Optional configuration for the HTML report.
        **kwargs: Additional keyword arguments passed to HTMLReporter.

    Returns:
        Path to the written file.

    Raises:
        ImportError: If jinja2 is not installed.

    Example:
        >>> from truthound.api import check
        >>> from truthound.html_reporter import write_html_report
        >>> report = check("data.csv")
        >>> path = write_html_report(report, "report.html")
        >>> print(f"Report written to {path}")
    """
    html = generate_html_report(report, config=config, **kwargs)
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_html_from_validation_result(
    result: Any,  # ValidationResult type
    config: HTMLReportConfig | None = None,
    **kwargs: Any,
) -> str:
    """Generate an HTML report directly from a ValidationResult.

    This function is useful when you already have a ValidationResult object
    and want to generate an HTML report without going through the Report
    intermediate.

    Args:
        result: The ValidationResult object.
        config: Optional configuration for the HTML report.
        **kwargs: Additional keyword arguments passed to HTMLReporter.

    Returns:
        HTML string representation of the result.

    Raises:
        ImportError: If jinja2 is not installed.
    """
    _require_jinja2()

    from truthound.reporters.html_reporter import HTMLReporter

    # Apply config if provided
    if config:
        kwargs.setdefault("title", config.title)
        kwargs.setdefault("theme", config.theme)
        kwargs.setdefault("custom_css", config.custom_css)
        kwargs.setdefault("include_metadata", config.include_metadata)
        kwargs.setdefault("include_statistics", config.include_statistics)

    reporter = HTMLReporter(**kwargs)
    return reporter.render(result)


# PII Report HTML Template
PII_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary: #3b82f6;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            color: var(--text-muted);
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
        }

        .status-badge.clean {
            background: var(--success);
            color: white;
        }

        .status-badge.detected {
            background: var(--warning);
            color: white;
        }

        .card {
            background: var(--card-bg);
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }

        .stat-item {
            text-align: center;
            padding: 1rem;
            background: var(--bg);
            border-radius: 0.375rem;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg);
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        tr:hover {
            background: var(--bg);
        }

        .column-name {
            font-family: monospace;
            font-size: 0.875rem;
            color: var(--primary);
        }

        .pii-type {
            font-weight: 500;
        }

        .confidence {
            font-family: monospace;
        }

        .confidence-high {
            color: var(--danger);
            font-weight: 600;
        }

        .confidence-medium {
            color: var(--warning);
        }

        .confidence-low {
            color: var(--text-muted);
        }

        .footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        .footer a {
            color: var(--primary);
            text-decoration: none;
        }

        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            table {
                font-size: 0.875rem;
            }

            th, td {
                padding: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p class="subtitle">{{ source }}</p>
            <span class="status-badge {{ 'detected' if findings else 'clean' }}">
                {{ 'PII Detected' if findings else 'No PII Found' }}
            </span>
        </div>

        <div class="card">
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{{ "{:,}".format(row_count) }}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ findings|length }}</div>
                    <div class="stat-label">PII Columns</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ pii_types|length }}</div>
                    <div class="stat-label">PII Types</div>
                </div>
            </div>
        </div>

        {% if findings %}
        <div class="card">
            <h2>PII Findings ({{ findings|length }})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>PII Type</th>
                        <th>Count</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {% for finding in findings %}
                    <tr>
                        <td class="column-name">{{ finding.column }}</td>
                        <td class="pii-type">{{ finding.pii_type }}</td>
                        <td>{{ "{:,}".format(finding.count) }}</td>
                        <td class="confidence {% if finding.confidence >= 90 %}confidence-high{% elif finding.confidence >= 70 %}confidence-medium{% else %}confidence-low{% endif %}">
                            {{ finding.confidence }}%
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="footer">
            <p>
                Generated by <a href="https://github.com/seadonggyun4/Truthound">Truthound</a>
                at {{ generated_at }}
            </p>
        </div>
    </div>
</body>
</html>
"""


def generate_pii_html_report(
    pii_report: "PIIReport",
    title: str = "Truthound PII Scan Report",
    **kwargs: Any,
) -> str:
    """Generate an HTML report from a PIIReport object.

    Args:
        pii_report: The PIIReport object containing PII findings.
        title: Title for the HTML report.
        **kwargs: Additional keyword arguments (for future extensibility).

    Returns:
        HTML string representation of the PII report.

    Raises:
        ImportError: If jinja2 is not installed.

    Example:
        >>> from truthound.api import scan
        >>> from truthound.html_reporter import generate_pii_html_report
        >>> pii_report = scan("data.csv")
        >>> html = generate_pii_html_report(pii_report)
        >>> with open("pii_report.html", "w") as f:
        ...     f.write(html)
    """
    _require_jinja2()

    from datetime import datetime

    from jinja2 import Template

    template = Template(PII_REPORT_TEMPLATE)

    # Extract unique PII types
    pii_types = set()
    for finding in pii_report.findings:
        pii_types.add(finding.get("pii_type", "unknown"))

    # Convert findings to objects with attribute access for template
    class Finding:
        def __init__(self, data: dict[str, Any]) -> None:
            self.column = data.get("column", "")
            self.pii_type = data.get("pii_type", "")
            self.count = data.get("count", 0)
            self.confidence = data.get("confidence", 0)

    findings = [Finding(f) for f in pii_report.findings]

    html = template.render(
        title=title,
        source=pii_report.source,
        row_count=pii_report.row_count,
        findings=findings,
        pii_types=pii_types,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    return html


def write_pii_html_report(
    pii_report: "PIIReport",
    output_path: str | Path,
    title: str = "Truthound PII Scan Report",
    **kwargs: Any,
) -> Path:
    """Generate and write a PII HTML report to a file.

    Args:
        pii_report: The PIIReport object containing PII findings.
        output_path: Path where the HTML file should be written.
        title: Title for the HTML report.
        **kwargs: Additional keyword arguments.

    Returns:
        Path to the written file.

    Raises:
        ImportError: If jinja2 is not installed.
    """
    html = generate_pii_html_report(pii_report, title=title, **kwargs)
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# Export public API
__all__ = [
    "generate_html_report",
    "write_html_report",
    "generate_html_from_validation_result",
    "generate_pii_html_report",
    "write_pii_html_report",
    "HTMLReportConfig",
    "HAS_JINJA2",
]
