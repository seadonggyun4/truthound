"""HTML format reporter.

This module provides a reporter that outputs validation results in HTML format.
Requires the jinja2 package.

Install with: pip install truthound[all]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Lazy import to avoid ImportError when jinja2 is not installed
try:
    from jinja2 import Environment, PackageLoader, FileSystemLoader, select_autoescape

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

from truthound.reporters.base import (
    ReporterConfig,
    RenderError,
    ValidationReporter,
)

if TYPE_CHECKING:
    from truthound.reporters._protocols import Jinja2EnvironmentProtocol
    from truthound.stores.results import ValidationResult


def _require_jinja2() -> None:
    """Check if jinja2 is available."""
    if not HAS_JINJA2:
        raise ImportError(
            "jinja2 is required for HTMLReporter. "
            "Install with: pip install truthound[all]"
        )


# Default HTML template
DEFAULT_TEMPLATE = """
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
            --critical: #dc2626;
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

        .status-badge.success {
            background: var(--success);
            color: white;
        }

        .status-badge.failure {
            background: var(--danger);
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

        .severity-bar {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .severity-item {
            display: flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
        }

        .severity-critical {
            background: #fee2e2;
            color: var(--critical);
        }

        .severity-high {
            background: #fef3c7;
            color: #d97706;
        }

        .severity-medium {
            background: #fef9c3;
            color: #ca8a04;
        }

        .severity-low {
            background: #f1f5f9;
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

        .issue-type {
            font-weight: 500;
        }

        .count {
            font-family: monospace;
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
            <p class="subtitle">{{ result.data_asset }}</p>
            <span class="status-badge {{ 'success' if result.success else 'failure' }}">
                {{ 'Passed' if result.success else 'Failed' }}
            </span>
        </div>

        <div class="card">
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{{ "{:,}".format(statistics.total_rows) }}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ statistics.total_columns }}</div>
                    <div class="stat-label">Columns</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ statistics.total_issues }}</div>
                    <div class="stat-label">Issues</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(statistics.pass_rate * 100) }}%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
            </div>

            {% if statistics.total_issues > 0 %}
            <div class="severity-bar">
                {% if statistics.critical_issues > 0 %}
                <span class="severity-item severity-critical">
                    {{ statistics.critical_issues }} Critical
                </span>
                {% endif %}
                {% if statistics.high_issues > 0 %}
                <span class="severity-item severity-high">
                    {{ statistics.high_issues }} High
                </span>
                {% endif %}
                {% if statistics.medium_issues > 0 %}
                <span class="severity-item severity-medium">
                    {{ statistics.medium_issues }} Medium
                </span>
                {% endif %}
                {% if statistics.low_issues > 0 %}
                <span class="severity-item severity-low">
                    {{ statistics.low_issues }} Low
                </span>
                {% endif %}
            </div>
            {% endif %}
        </div>

        {% if issues %}
        <div class="card">
            <h2>Issues ({{ issues|length }})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Issue Type</th>
                        <th>Count</th>
                        <th>Severity</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {% for issue in issues %}
                    <tr>
                        <td class="column-name">{{ issue.column or '-' }}</td>
                        <td class="issue-type">{{ issue.issue_type or issue.validator_name }}</td>
                        <td class="count">{{ "{:,}".format(issue.count) }}</td>
                        <td>
                            <span class="severity-item severity-{{ issue.severity|lower if issue.severity else 'low' }}">
                                {{ issue.severity or 'low' }}
                            </span>
                        </td>
                        <td>{{ issue.message[:80] ~ '...' if issue.message and issue.message|length > 80 else issue.message or '-' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="card">
            <h2>Details</h2>
            <table>
                <tr>
                    <th>Run ID</th>
                    <td><code>{{ result.run_id }}</code></td>
                </tr>
                <tr>
                    <th>Run Time</th>
                    <td>{{ result.run_time.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                </tr>
                {% if result.suite_name %}
                <tr>
                    <th>Suite Name</th>
                    <td>{{ result.suite_name }}</td>
                </tr>
                {% endif %}
                {% if statistics.execution_time_ms > 0 %}
                <tr>
                    <th>Execution Time</th>
                    <td>{{ "%.2f"|format(statistics.execution_time_ms / 1000) }}s</td>
                </tr>
                {% endif %}
            </table>
        </div>

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


@dataclass
class HTMLReporterConfig(ReporterConfig):
    """Configuration for HTML reporter.

    Attributes:
        template_path: Path to custom Jinja2 template file.
        inline_css: Whether to inline CSS (vs. external stylesheet).
        theme: Color theme ("light" or "dark").
        include_charts: Whether to include visualizations (future).
        custom_css: Additional CSS to include.
    """

    template_path: str | None = None
    inline_css: bool = True
    theme: str = "light"
    include_charts: bool = False
    custom_css: str = ""


class HTMLReporter(ValidationReporter[HTMLReporterConfig]):
    """HTML format reporter for validation results.

    Outputs validation results as a standalone HTML page with embedded
    styling. Can use custom Jinja2 templates for full customization.

    Example:
        >>> reporter = HTMLReporter(title="My Report")
        >>> html = reporter.render(validation_result)
        >>> reporter.write(validation_result, "report.html")
    """

    name = "html"
    file_extension = ".html"
    content_type = "text/html"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the HTML reporter.

        Args:
            **kwargs: Configuration options.

        Raises:
            ImportError: If jinja2 is not installed.
        """
        _require_jinja2()
        super().__init__(**kwargs)
        self._env: Jinja2EnvironmentProtocol | None = None

    @classmethod
    def _default_config(cls) -> HTMLReporterConfig:
        """Create default configuration."""
        return HTMLReporterConfig()

    def _get_environment(self) -> "Jinja2EnvironmentProtocol":
        """Get or create the Jinja2 environment.

        Returns:
            Configured Jinja2 Environment.
        """
        if self._env is None:
            _require_jinja2()

            if self._config.template_path:
                # Use custom template from file
                template_dir = Path(self._config.template_path).parent
                self._env = Environment(
                    loader=FileSystemLoader(template_dir),
                    autoescape=select_autoescape(["html", "xml"]),
                )
            else:
                # Use default template
                from jinja2 import BaseLoader

                class StringLoader(BaseLoader):
                    def get_source(self, environment: Any, template: str) -> tuple[str, str | None, Any]:
                        return DEFAULT_TEMPLATE, None, lambda: False

                self._env = Environment(
                    loader=StringLoader(),
                    autoescape=select_autoescape(["html", "xml"]),
                )

        return self._env

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as HTML.

        Args:
            data: The validation result to render.

        Returns:
            HTML string representation.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            env = self._get_environment()

            if self._config.template_path:
                template_name = Path(self._config.template_path).name
                template = env.get_template(template_name)
            else:
                template = env.get_template("default")

            # Prepare issues list
            issues = [r for r in data.results if not r.success]

            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_issues = sorted(
                issues,
                key=lambda x: severity_order.get(
                    x.severity.lower() if x.severity else "low", 4
                ),
            )

            # Render template
            html = template.render(
                title=self._config.title,
                result=data,
                statistics=data.statistics,
                issues=sorted_issues,
                generated_at=datetime.now().strftime(self._config.timestamp_format),
                config=self._config,
            )

            # Add custom CSS if specified
            if self._config.custom_css:
                custom_style = f"<style>{self._config.custom_css}</style>"
                html = html.replace("</head>", f"{custom_style}</head>")

            return html

        except Exception as e:
            raise RenderError(f"Failed to render HTML: {e}")
