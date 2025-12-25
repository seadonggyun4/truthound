"""Update data documentation action.

This action updates data quality documentation sites with the latest
validation results, creating reports and dashboards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class UpdateDocsConfig(ActionConfig):
    """Configuration for update docs action.

    Attributes:
        site_path: Path to documentation site root.
        site_name: Name of the documentation site.
        format: Documentation format ("html", "markdown", "json").
        template_path: Custom template path for report generation.
        include_history: Include historical trend charts.
        max_history_items: Maximum history items to show.
        auto_open: Auto-open documentation in browser after update.
    """

    site_path: str | Path = "./truthound_docs"
    site_name: str = "Truthound Data Docs"
    format: str = "html"
    template_path: str | Path | None = None
    include_history: bool = True
    max_history_items: int = 50
    auto_open: bool = False
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class UpdateDataDocs(BaseAction[UpdateDocsConfig]):
    """Action to update data quality documentation.

    This action generates and updates documentation sites with
    validation results, including reports, charts, and history.

    Example:
        >>> action = UpdateDataDocs(
        ...     site_path="./docs",
        ...     site_name="Data Quality Portal",
        ...     include_history=True,
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "update_docs"

    @classmethod
    def _default_config(cls) -> UpdateDocsConfig:
        return UpdateDocsConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Update the documentation site."""
        config = self._config
        site_path = Path(config.site_path)

        # Create site structure
        site_path.mkdir(parents=True, exist_ok=True)
        (site_path / "runs").mkdir(exist_ok=True)
        (site_path / "assets").mkdir(exist_ok=True)

        # Generate documentation files
        generated_files: list[str] = []

        # Generate run report
        run_report_path = self._generate_run_report(site_path, checkpoint_result)
        generated_files.append(str(run_report_path))

        # Update index
        index_path = self._update_index(site_path, checkpoint_result)
        generated_files.append(str(index_path))

        # Update history
        if config.include_history:
            history_path = self._update_history(site_path, checkpoint_result)
            generated_files.append(str(history_path))

        # Generate assets (CSS, JS)
        self._generate_assets(site_path)

        # Auto-open in browser if configured
        if config.auto_open:
            import webbrowser
            webbrowser.open(f"file://{index_path.absolute()}")

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message=f"Documentation updated at {site_path}",
            details={
                "site_path": str(site_path),
                "generated_files": generated_files,
                "format": config.format,
            },
        )

    def _generate_run_report(
        self,
        site_path: Path,
        checkpoint_result: "CheckpointResult",
    ) -> Path:
        """Generate report for a specific run."""
        config = self._config
        run_id = checkpoint_result.run_id
        run_path = site_path / "runs" / f"{run_id}.{config.format}"

        if config.format == "html":
            content = self._generate_html_report(checkpoint_result)
        elif config.format == "markdown":
            content = self._generate_markdown_report(checkpoint_result)
        else:
            content = json.dumps(checkpoint_result.to_dict(), indent=2, default=str)

        run_path.write_text(content)
        return run_path

    def _generate_html_report(self, checkpoint_result: "CheckpointResult") -> str:
        """Generate HTML report for a run."""
        status = checkpoint_result.status.value
        status_color = {
            "success": "#28a745",
            "failure": "#dc3545",
            "error": "#dc3545",
            "warning": "#ffc107",
        }.get(status, "#6c757d")

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Run Report - {checkpoint_result.run_id}</title>
    <link rel="stylesheet" href="../assets/style.css">
</head>
<body>
    <nav><a href="../index.html">← Back to Index</a></nav>
    <main>
        <header>
            <h1>Validation Run Report</h1>
            <div class="meta">
                <span class="run-id">Run ID: {checkpoint_result.run_id}</span>
                <span class="checkpoint">Checkpoint: {checkpoint_result.checkpoint_name}</span>
                <span class="time">{checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>

        <section class="summary">
            <h2>Summary</h2>
            <div class="status-badge" style="background-color: {status_color}">
                {status.upper()}
            </div>
            <div class="stats-grid">
                <div class="stat">
                    <span class="label">Total Issues</span>
                    <span class="value">{stats.total_issues if stats else 0}</span>
                </div>
                <div class="stat">
                    <span class="label">Critical</span>
                    <span class="value critical">{stats.critical_issues if stats else 0}</span>
                </div>
                <div class="stat">
                    <span class="label">High</span>
                    <span class="value high">{stats.high_issues if stats else 0}</span>
                </div>
                <div class="stat">
                    <span class="label">Medium</span>
                    <span class="value medium">{stats.medium_issues if stats else 0}</span>
                </div>
                <div class="stat">
                    <span class="label">Low</span>
                    <span class="value low">{stats.low_issues if stats else 0}</span>
                </div>
                <div class="stat">
                    <span class="label">Pass Rate</span>
                    <span class="value">{stats.pass_rate * 100:.1f}%</span>
                </div>
            </div>
        </section>

        <section class="data-info">
            <h2>Data Information</h2>
            <table>
                <tr><th>Data Asset</th><td>{checkpoint_result.data_asset}</td></tr>
                <tr><th>Rows</th><td>{stats.total_rows if stats else 'N/A':,}</td></tr>
                <tr><th>Columns</th><td>{stats.total_columns if stats else 'N/A'}</td></tr>
                <tr><th>Execution Time</th><td>{stats.execution_time_ms if stats else 0:.2f} ms</td></tr>
            </table>
        </section>

        <section class="actions">
            <h2>Actions Executed</h2>
            <ul class="action-list">
                {"".join(f'<li class="{a.status.value}">{a.action_name}: {a.message}</li>' for a in checkpoint_result.action_results)}
            </ul>
        </section>
    </main>
    <footer>
        <p>Generated by Truthound on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>"""
        return html

    def _generate_markdown_report(self, checkpoint_result: "CheckpointResult") -> str:
        """Generate Markdown report for a run."""
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        md = f"""# Validation Run Report

**Run ID:** {checkpoint_result.run_id}
**Checkpoint:** {checkpoint_result.checkpoint_name}
**Time:** {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}
**Status:** {checkpoint_result.status.value.upper()}

## Summary

| Metric | Value |
|--------|-------|
| Total Issues | {stats.total_issues if stats else 0} |
| Critical | {stats.critical_issues if stats else 0} |
| High | {stats.high_issues if stats else 0} |
| Medium | {stats.medium_issues if stats else 0} |
| Low | {stats.low_issues if stats else 0} |
| Pass Rate | {stats.pass_rate * 100 if stats else 100:.1f}% |

## Data Information

| Property | Value |
|----------|-------|
| Data Asset | {checkpoint_result.data_asset} |
| Rows | {stats.total_rows if stats else 'N/A':,} |
| Columns | {stats.total_columns if stats else 'N/A'} |
| Execution Time | {stats.execution_time_ms if stats else 0:.2f} ms |

## Actions Executed

"""
        for action in checkpoint_result.action_results:
            md += f"- **{action.action_name}** ({action.status.value}): {action.message}\n"

        md += f"\n---\n*Generated by Truthound on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        return md

    def _update_index(
        self,
        site_path: Path,
        checkpoint_result: "CheckpointResult",
    ) -> Path:
        """Update the main index page."""
        config = self._config
        index_path = site_path / f"index.{config.format}"

        # Load existing runs
        runs_dir = site_path / "runs"
        runs = []
        for run_file in runs_dir.glob(f"*.{config.format}"):
            run_id = run_file.stem
            # Extract run info from filename or parse file
            runs.append({
                "id": run_id,
                "file": f"runs/{run_file.name}",
            })

        # Sort by run_id (which contains timestamp)
        runs.sort(key=lambda x: x["id"], reverse=True)
        runs = runs[:config.max_history_items]

        if config.format == "html":
            content = self._generate_html_index(checkpoint_result, runs)
        elif config.format == "markdown":
            content = self._generate_markdown_index(checkpoint_result, runs)
        else:
            content = json.dumps({"latest": checkpoint_result.to_dict(), "runs": runs}, indent=2, default=str)

        index_path.write_text(content)
        return index_path

    def _generate_html_index(
        self,
        latest: "CheckpointResult",
        runs: list[dict[str, Any]],
    ) -> str:
        """Generate HTML index page."""
        config = self._config
        status = latest.status.value
        status_color = {
            "success": "#28a745",
            "failure": "#dc3545",
            "error": "#dc3545",
            "warning": "#ffc107",
        }.get(status, "#6c757d")

        runs_html = "\n".join(
            f'<li><a href="{r["file"]}">{r["id"]}</a></li>'
            for r in runs
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.site_name}</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <header class="main-header">
        <h1>{config.site_name}</h1>
        <p>Data Quality Documentation</p>
    </header>

    <main>
        <section class="latest">
            <h2>Latest Run</h2>
            <div class="latest-card">
                <div class="status-badge" style="background-color: {status_color}">
                    {status.upper()}
                </div>
                <p><strong>Run ID:</strong> <a href="runs/{latest.run_id}.html">{latest.run_id}</a></p>
                <p><strong>Checkpoint:</strong> {latest.checkpoint_name}</p>
                <p><strong>Time:</strong> {latest.run_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </section>

        <section class="history">
            <h2>Run History</h2>
            <ul class="runs-list">
                {runs_html}
            </ul>
        </section>
    </main>

    <footer>
        <p>Generated by Truthound on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>"""
        return html

    def _generate_markdown_index(
        self,
        latest: "CheckpointResult",
        runs: list[dict[str, Any]],
    ) -> str:
        """Generate Markdown index page."""
        config = self._config

        runs_md = "\n".join(f"- [{r['id']}]({r['file']})" for r in runs)

        md = f"""# {config.site_name}

## Latest Run

- **Status:** {latest.status.value.upper()}
- **Run ID:** [{latest.run_id}](runs/{latest.run_id}.md)
- **Checkpoint:** {latest.checkpoint_name}
- **Time:** {latest.run_time.strftime('%Y-%m-%d %H:%M:%S')}

## Run History

{runs_md}

---
*Generated by Truthound on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return md

    def _update_history(
        self,
        site_path: Path,
        checkpoint_result: "CheckpointResult",
    ) -> Path:
        """Update history data file for charts."""
        history_path = site_path / "history.json"

        # Load existing history
        history: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                history = []

        # Add new entry
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        history.append({
            "run_id": checkpoint_result.run_id,
            "checkpoint": checkpoint_result.checkpoint_name,
            "run_time": checkpoint_result.run_time.isoformat(),
            "status": checkpoint_result.status.value,
            "total_issues": stats.total_issues if stats else 0,
            "critical_issues": stats.critical_issues if stats else 0,
            "high_issues": stats.high_issues if stats else 0,
            "pass_rate": stats.pass_rate if stats else 1.0,
        })

        # Keep only recent entries
        history = history[-self._config.max_history_items:]

        history_path.write_text(json.dumps(history, indent=2))
        return history_path

    def _generate_assets(self, site_path: Path) -> None:
        """Generate static assets (CSS, JS)."""
        assets_path = site_path / "assets"
        assets_path.mkdir(exist_ok=True)

        # Generate CSS
        css = """/* Truthound Data Docs Styles */
:root {
    --color-success: #28a745;
    --color-warning: #ffc107;
    --color-danger: #dc3545;
    --color-info: #17a2b8;
    --color-bg: #f8f9fa;
    --color-text: #212529;
    --color-border: #dee2e6;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--color-text);
    background: var(--color-bg);
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

header, nav { margin-bottom: 20px; }
nav a { color: var(--color-info); text-decoration: none; }
nav a:hover { text-decoration: underline; }

h1, h2, h3 { margin-bottom: 15px; }
h1 { font-size: 2em; }
h2 { font-size: 1.5em; border-bottom: 1px solid var(--color-border); padding-bottom: 10px; }

.status-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    color: white;
    font-weight: bold;
    text-transform: uppercase;
    margin: 10px 0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.stat {
    background: white;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat .label { display: block; font-size: 0.9em; color: #666; }
.stat .value { font-size: 1.5em; font-weight: bold; }
.stat .value.critical { color: var(--color-danger); }
.stat .value.high { color: #e67e22; }
.stat .value.medium { color: var(--color-warning); }
.stat .value.low { color: var(--color-info); }

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 15px 0;
}

th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--color-border); }
th { background: #f1f3f5; font-weight: 600; }

.action-list { list-style: none; }
.action-list li {
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 5px;
    background: white;
}
.action-list li.success { border-left: 4px solid var(--color-success); }
.action-list li.failure, .action-list li.error { border-left: 4px solid var(--color-danger); }
.action-list li.skipped { border-left: 4px solid #6c757d; }

.latest-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.runs-list { list-style: none; }
.runs-list li { padding: 10px 0; border-bottom: 1px solid var(--color-border); }
.runs-list a { color: var(--color-info); text-decoration: none; }
.runs-list a:hover { text-decoration: underline; }

footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--color-border);
    color: #666;
    font-size: 0.9em;
}
"""
        (assets_path / "style.css").write_text(css)

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if self._config.format not in ("html", "markdown", "json"):
            errors.append(f"Invalid format: {self._config.format}")

        return errors
