"""Validation-focused Data Docs builders."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from truthound.datadocs.base import ReportTheme
from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.styles import get_complete_stylesheet
from truthound.datadocs.themes import get_theme
from truthound.reporters.adapters import canonicalize_validation_run_result
from truthound.reporters.presentation import RunPresentation, build_run_presentation


class ValidationDataConverter:
    """Convert ValidationRunResult into shared report context data."""

    def __init__(self, result: Any, *, title: str = "Truthound Validation Data Docs") -> None:
        self.run_result = canonicalize_validation_run_result(result, warn_legacy=True)
        self.presentation = build_run_presentation(self.run_result, title=title)

    def build_context(
        self,
        *,
        title: str = "Truthound Validation Data Docs",
        subtitle: str = "",
        theme: str = "professional",
        locale: str = "en",
    ) -> ReportContext:
        """Build the shared immutable ReportContext."""
        sections = {
            "overview": self._overview_section(),
            "checks": self._checks_section(),
            "issues": self._issues_section(),
            "execution_issues": self._execution_issues_section(),
            "metadata": self._metadata_section(),
        }
        data = ReportData(
            raw=self.run_result.to_dict(),
            sections=sections,
            metadata={
                "title": title,
                "subtitle": subtitle,
                "source": self.presentation.source,
                "suite_name": self.presentation.suite_name,
                "run_id": self.presentation.run_id,
            },
            alerts=self._alerts(),
        )
        return ReportContext(
            data=data,
            locale=locale,
            theme=theme,
            output_format="html",
            options={"presentation": self.presentation},
        )

    def _overview_section(self) -> dict[str, Any]:
        summary = self.presentation.summary
        return {
            "metrics": [
                ("Status", self.presentation.status.upper()),
                ("Rows", f"{summary.total_rows:,}"),
                ("Columns", f"{summary.total_columns:,}"),
                ("Checks", str(summary.total_checks)),
                ("Issues", str(summary.total_issues)),
                ("Pass Rate", f"{summary.pass_rate:.1%}"),
            ],
            "severity": dict(self.presentation.issue_counts_by_severity),
        }

    def _checks_section(self) -> dict[str, Any]:
        rows = []
        for check in self.presentation.checks:
            rows.append({
                "name": check.name,
                "category": check.category,
                "status": "passed" if check.success else "failed",
                "issue_count": check.issue_count,
                "top_severity": check.top_severity or "-",
                "columns": ", ".join(check.columns) if check.columns else "-",
            })
        return {"rows": rows}

    def _issues_section(self) -> dict[str, Any]:
        rows = []
        for issue in self.presentation.issues:
            rows.append({
                "validator": issue.validator_name,
                "column": issue.column or "-",
                "issue_type": issue.issue_type,
                "count": issue.count,
                "severity": issue.severity,
                "message": issue.message or "-",
            })
        return {"rows": rows}

    def _execution_issues_section(self) -> dict[str, Any]:
        rows = []
        for issue in self.presentation.execution_issues:
            rows.append({
                "check_name": issue.check_name,
                "message": issue.message,
                "exception_type": issue.exception_type or "-",
                "failure_category": issue.failure_category or "-",
                "retry_count": issue.retry_count,
            })
        return {"rows": rows}

    def _metadata_section(self) -> dict[str, Any]:
        runtime_environment = self.presentation.metadata.get("runtime_environment", {})
        return {
            "rows": [
                ("Run ID", self.presentation.run_id),
                ("Run Time", self.presentation.run_time.isoformat()),
                ("Suite", self.presentation.suite_name),
                ("Source", self.presentation.source),
                ("Execution Mode", self.presentation.execution_mode),
                ("Planned Execution Mode", self.presentation.planned_execution_mode),
                ("Result Format", self.presentation.result_format),
                ("Execution Issues", str(self.presentation.summary.total_execution_issues)),
                ("Runtime Environment", ", ".join(f"{k}={v}" for k, v in runtime_environment.items()) or "-"),
            ]
        }

    def _alerts(self) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        if self.presentation.summary.total_execution_issues:
            alerts.append({
                "title": "Execution issues detected",
                "message": f"{self.presentation.summary.total_execution_issues} execution issue(s) occurred during validation.",
                "severity": "error",
            })
        elif self.presentation.summary.total_issues:
            alerts.append({
                "title": "Validation issues detected",
                "message": f"{self.presentation.summary.total_issues} validation issue(s) require review.",
                "severity": "warning",
            })
        return alerts


class ValidationDocsBuilder:
    """Build validation Data Docs from ValidationRunResult."""

    def __init__(self, theme: ReportTheme | str = ReportTheme.PROFESSIONAL) -> None:
        self.theme = ReportTheme(theme) if isinstance(theme, str) else theme
        self._theme_config = get_theme(self.theme)

    def build(
        self,
        result: Any,
        *,
        title: str = "Truthound Validation Data Docs",
        subtitle: str = "",
    ) -> str:
        converter = ValidationDataConverter(result, title=title)
        context = converter.build_context(
            title=title,
            subtitle=subtitle,
            theme=self.theme.value,
        )
        return self._render_html(context)

    def save(self, html_content: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding="utf-8")
        return path

    def _render_html(self, context: ReportContext) -> str:
        presentation = self._get_presentation(context)
        sections = context.data.sections
        css = get_complete_stylesheet(
            self._theme_config.to_css_vars(),
            is_dark=self.theme == ReportTheme.DARK,
        )
        overview = sections["overview"]
        checks = sections["checks"]["rows"]
        issues = sections["issues"]["rows"]
        execution_issues = sections["execution_issues"]["rows"]
        metadata = sections["metadata"]["rows"]
        subtitle = context.subtitle or context.metadata.get("source", "")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(context.title)}</title>
  <style>
{css}
body {{
  background: var(--color-background);
  color: var(--color-text-primary);
  font-family: var(--font-family-sans);
  margin: 0;
  padding: 2rem;
}}
.container {{
  max-width: 1200px;
  margin: 0 auto;
}}
.hero, .panel {{
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-md);
  margin-bottom: 1.5rem;
  padding: 1.5rem;
}}
.hero h1 {{
  margin: 0 0 0.5rem;
}}
.muted {{
  color: var(--color-text-secondary);
}}
.badge {{
  border-radius: 999px;
  display: inline-block;
  font-size: 0.875rem;
  font-weight: 700;
  margin-top: 0.75rem;
  padding: 0.35rem 0.75rem;
}}
.badge.success {{
  background: rgba(34, 197, 94, 0.15);
  color: #166534;
}}
.badge.failure {{
  background: rgba(239, 68, 68, 0.15);
  color: #991b1b;
}}
.metrics {{
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}}
.metric {{
  background: var(--color-background-secondary);
  border-radius: var(--border-radius-md);
  padding: 1rem;
}}
.metric-label {{
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}}
.metric-value {{
  font-size: 1.4rem;
  font-weight: 700;
  margin-top: 0.25rem;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
th, td {{
  border-bottom: 1px solid var(--color-border);
  padding: 0.75rem;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  text-transform: uppercase;
}}
.severity-bar {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}}
.severity-pill {{
  background: var(--color-background-secondary);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
}}
.alerts {{
  display: grid;
  gap: 0.75rem;
  margin-bottom: 1rem;
}}
.alert {{
  background: rgba(245, 158, 11, 0.12);
  border-left: 4px solid #f59e0b;
  border-radius: var(--border-radius-md);
  padding: 1rem;
}}
code {{
  font-family: var(--font-family-mono);
}}
@media (max-width: 768px) {{
  body {{
    padding: 1rem;
  }}
  table {{
    font-size: 0.9rem;
  }}
}}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>{html.escape(context.title)}</h1>
      <p class="muted">{html.escape(subtitle)}</p>
      <span class="badge {'success' if presentation.success else 'failure'}">{presentation.status.upper()}</span>
    </section>
    {self._render_alerts(context.data.alerts)}
    <section class="panel">
      <h2>Overview</h2>
      <div class="metrics">
        {''.join(self._render_metric(label, value) for label, value in overview['metrics'])}
      </div>
      <div class="severity-bar">
        {''.join(self._render_severity_pill(name, count) for name, count in overview['severity'].items())}
      </div>
    </section>
    <section class="panel">
      <h2>Checks</h2>
      {self._render_table(["Check", "Category", "Status", "Issue Count", "Top Severity", "Columns"], checks)}
    </section>
    <section class="panel">
      <h2>Issues</h2>
      {self._render_table(["Validator", "Column", "Issue Type", "Count", "Severity", "Message"], issues)}
    </section>
    <section class="panel">
      <h2>Execution Issues</h2>
      {self._render_table(["Check", "Message", "Exception Type", "Failure Category", "Retries"], execution_issues)}
    </section>
    <section class="panel">
      <h2>Metadata</h2>
      <table>
        <tbody>
          {''.join(f"<tr><th>{html.escape(label)}</th><td><code>{html.escape(value)}</code></td></tr>" for label, value in metadata)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>"""

    def _get_presentation(self, context: ReportContext) -> RunPresentation:
        presentation = context.options.get("presentation")
        if not isinstance(presentation, RunPresentation):
            raise TypeError("Validation report context is missing RunPresentation data.")
        return presentation

    @staticmethod
    def _render_metric(label: str, value: str) -> str:
        return (
            '<div class="metric">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div>'
            '</div>'
        )

    @staticmethod
    def _render_severity_pill(name: str, count: int) -> str:
        return f'<span class="severity-pill">{html.escape(name.title())}: {count}</span>'

    @staticmethod
    def _render_alerts(alerts: list[dict[str, Any]]) -> str:
        if not alerts:
            return ""
        body = "".join(
            '<div class="alert">'
            f'<strong>{html.escape(str(alert.get("title", "")))}</strong><br>'
            f'{html.escape(str(alert.get("message", "")))}'
            '</div>'
            for alert in alerts
        )
        return f'<section class="alerts">{body}</section>'

    @staticmethod
    def _render_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
        if not rows:
            return '<p class="muted">No data available.</p>'
        keys = [header.lower().replace(" ", "_") for header in headers]
        header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        row_html = []
        for row in rows:
            row_html.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(row.get(key, '-')))}</td>"
                    for key in keys
                )
                + "</tr>"
            )
        return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def generate_validation_report(
    result: Any,
    *,
    title: str = "Truthound Validation Data Docs",
    subtitle: str = "",
    theme: ReportTheme | str = ReportTheme.PROFESSIONAL,
    output_path: str | Path | None = None,
) -> str:
    """Generate static HTML validation Data Docs."""
    builder = ValidationDocsBuilder(theme=theme)
    html_content = builder.build(result, title=title, subtitle=subtitle)
    if output_path:
        builder.save(html_content, output_path)
    return html_content
