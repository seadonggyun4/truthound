"""Validation-focused Data Docs builders."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from truthound.datadocs.base import ReportTheme
from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.i18n import ReportCatalog, get_catalog
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
        theme: str = "light",
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
            alerts=self._alerts(locale),
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

    def _alerts(self, locale: str = "en") -> list[dict[str, Any]]:
        catalog = get_catalog(locale)
        alerts: list[dict[str, Any]] = []
        if self.presentation.summary.total_execution_issues:
            alerts.append({
                "title": catalog.get("validation.execution_title", default="Execution issues detected"),
                "message": catalog.get("validation.execution_message", default="{count} execution issue(s) occurred during validation.", count=self.presentation.summary.total_execution_issues),
                "severity": "error",
            })
        elif self.presentation.summary.total_issues:
            alerts.append({
                "title": catalog.get("validation.quality_title", default="Validation issues detected"),
                "message": catalog.get("validation.quality_message", default="{count} validation issue(s) require review.", count=self.presentation.summary.total_issues),
                "severity": "warning",
            })
        return alerts


class ValidationDocsBuilder:
    """Build validation Data Docs from ValidationRunResult."""

    def __init__(self, theme: ReportTheme | str = ReportTheme.LIGHT, *, locale: str = "en") -> None:
        if locale not in {"en", "ko"}:
            raise ValueError("Unsupported validation report locale; expected en or ko.")
        self.locale = locale
        self._theme_config = get_theme(theme)
        self.theme = self._theme_config.name

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
            theme=self.theme,
            locale=self.locale,
        )
        return self._render_html(context)

    def save(self, html_content: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding="utf-8")
        return path

    def _render_html(self, context: ReportContext) -> str:
        catalog = get_catalog(context.locale)
        def label(value: str) -> str:
            return catalog.get("validation.label." + value.lower().replace(" ", "_"), default=value)

        presentation = self._get_presentation(context)
        sections = context.data.sections
        css = get_complete_stylesheet(
            self._theme_config.to_css_vars(),
            is_dark=self.theme == ReportTheme.DARK.value,
        )
        overview = sections["overview"]
        checks = sections["checks"]["rows"]
        issues = sections["issues"]["rows"]
        execution_issues = sections["execution_issues"]["rows"]
        metadata = sections["metadata"]["rows"]
        subtitle = context.subtitle or context.metadata.get("source", "")

        return f"""<!DOCTYPE html>
<html lang="{html.escape(context.locale, quote=True)}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(context.title)}</title>
  <style>
{css}
body {{
  background: var(--color-background);
  color: var(--color-text-primary);
  font-family: var(--font-family);
  margin: 0;
  padding: 10mm 0;
}}
.container {{
  width: 210mm;
  max-width: 210mm;
  min-height: 297mm;
  margin: 0 auto;
  padding: 20mm 18mm 16mm;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  box-shadow: 0 2mm 10mm var(--color-shadow);
}}
.hero, .panel {{
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  margin-bottom: 9mm;
  padding: 0;
  page-break-inside: avoid;
  break-inside: avoid;
}}
.hero {{
  border-bottom: 2pt solid var(--color-primary);
  padding-bottom: 5mm;
}}
.hero h1 {{
  color: var(--color-primary);
  font-size: var(--font-size-3xl);
  margin: 0 0 2mm;
}}
.panel h2 {{
  color: var(--color-primary);
  font-size: var(--font-size-xl);
  border-bottom: 1.2pt solid var(--color-primary);
  padding-bottom: 2mm;
  margin-bottom: 4mm;
}}
.panel h2::before {{
  content: "□ ";
}}
.muted {{
  color: var(--color-text-secondary);
}}
.badge {{
  border-radius: var(--border-radius-sm);
  display: inline-block;
  font-size: var(--font-size-sm);
  font-weight: 700;
  margin-top: 3mm;
  padding: 1.5mm 3mm;
}}
.badge.success {{
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}}
.badge.failure {{
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-error);
}}
.metrics {{
  display: grid;
  gap: 3mm;
  grid-template-columns: repeat(auto-fit, minmax(34mm, 1fr));
  margin-bottom: 4mm;
}}
.metric {{
  background: rgba(31, 78, 121, 0.06);
  border: 0.6pt solid var(--color-border);
  border-radius: var(--border-radius-md);
  padding: 3mm;
}}
.metric-label {{
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
}}
.metric-value {{
  color: var(--color-primary);
  font-size: var(--font-size-xl);
  font-weight: 700;
  margin-top: 1mm;
}}
table {{
  border-top: 1pt solid var(--color-primary);
  border-left: 0.5pt solid var(--color-border);
  border-collapse: collapse;
  width: 100%;
  font-size: var(--font-size-sm);
}}
th, td {{
  border-right: 0.5pt solid var(--color-border);
  border-bottom: 0.5pt solid var(--color-border);
  padding: 2mm 2.5mm;
  text-align: left;
  vertical-align: top;
}}
th {{
  background: var(--color-primary);
  color: var(--color-background);
  font-size: var(--font-size-sm);
  font-weight: 700;
  text-align: center;
}}
.severity-bar {{
  display: flex;
  flex-wrap: wrap;
  gap: 2mm;
  margin-top: 3mm;
}}
.severity-pill {{
  background: var(--color-surface);
  border: 0.5pt solid var(--color-border);
  border-radius: var(--border-radius-sm);
  padding: 1.5mm 3mm;
}}
.alerts {{
  display: grid;
  gap: 3mm;
  margin-bottom: 6mm;
}}
.alert {{
  background: rgba(245, 158, 11, 0.12);
  border-left: 3pt solid var(--color-warning);
  border-radius: var(--border-radius-md);
  padding: 3mm 4mm;
  break-inside: avoid;
}}
code {{
  font-family: var(--font-family-mono);
}}
@page {{
  size: A4 portrait;
  margin: 16mm 14mm;
}}
@media print {{
  body {{
    background: white;
    padding: 0;
  }}
  .container {{
    width: auto;
    max-width: none;
    min-height: auto;
    padding: 0;
    border: 0;
    box-shadow: none;
  }}
  tr, thead, tfoot {{
    page-break-inside: avoid;
    break-inside: avoid;
  }}
}}
@media (max-width: 768px) {{
  body {{
    padding: 0;
  }}
  .container {{
    width: 100%;
    max-width: none;
    min-height: 0;
    padding: 1rem;
    border: 0;
  }}
  table {{
    font-size: var(--font-size-sm);
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
      <h2>{html.escape(label("Overview"))}</h2>
      <div class="metrics">
        {''.join(self._render_metric(label(name), value) for name, value in overview['metrics'])}
      </div>
      <div class="severity-bar">
        {''.join(self._render_severity_pill(label(name.title()), count) for name, count in overview['severity'].items())}
      </div>
    </section>
    <section class="panel">
      <h2>{html.escape(label("Checks"))}</h2>
      {self._render_table(["Check", "Category", "Status", "Issue Count", "Top Severity", "Columns"], checks, catalog=catalog, fields=["name", "category", "status", "issue_count", "top_severity", "columns"])}
    </section>
    <section class="panel">
      <h2>{html.escape(label("Issues"))}</h2>
      {self._render_table(["Validator", "Column", "Issue Type", "Count", "Severity", "Message"], issues, catalog=catalog)}
    </section>
    <section class="panel">
      <h2>{html.escape(label("Execution Issues"))}</h2>
      {self._render_table(["Check", "Message", "Exception Type", "Failure Category", "Retries"], execution_issues, catalog=catalog, fields=["check_name", "message", "exception_type", "failure_category", "retry_count"])}
    </section>
    <section class="panel">
      <h2>{html.escape(label("Metadata"))}</h2>
      <table>
        <tbody>
          {''.join(f"<tr><th>{html.escape(label(name))}</th><td><code>{html.escape(value)}</code></td></tr>" for name, value in metadata)}
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
    def _render_table(headers: list[str], rows: list[dict[str, Any]], *, catalog: ReportCatalog | None = None, fields: list[str] | None = None) -> str:
        if not rows:
            message = catalog.get("validation.no_data") if catalog else "No data available."
            return f'<p class="muted">{html.escape(message)}</p>'
        # Display labels are not always the document model's field names.
        keys = fields if fields is not None else [header.lower().replace(" ", "_") for header in headers]
        if len(keys) != len(headers):
            raise ValueError("Report table fields must match its headers")
        labels = [catalog.get("validation.label." + header.lower().replace(" ", "_"), default=header) if catalog else header for header in headers]
        header_html = "".join(f"<th>{html.escape(header)}</th>" for header in labels)
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
    theme: ReportTheme | str = ReportTheme.LIGHT,
    output_path: str | Path | None = None,
    locale: str = "en",
) -> str:
    """Generate validation Data Docs in English (default) or Korean.

    Locale changes display labels and alerts only; canonical quality values,
    machine statuses and caller-provided content are preserved.
    """
    builder = ValidationDocsBuilder(theme=theme, locale=locale)
    html_content = builder.build(result, title=title, subtitle=subtitle)
    if output_path:
        builder.save(html_content, output_path)
    return html_content
