# truthound new reporter

Create a custom reporter with boilerplate code.

## Synopsis

```bash
truthound new reporter <name> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Reporter name (snake_case) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--template` | `-t` | `basic` | Template type (basic, full) |
| `--author` | `-a` | None | Author name |
| `--description` | `-d` | None | Reporter description |
| `--tests/--no-tests` | | `--tests` | Generate test code |
| `--docs/--no-docs` | | `--no-docs` | Generate documentation |
| `--extension` | `-e` | `.txt` | Output file extension |
| `--content-type` | | `text/plain` | MIME Content-Type |

## Description

The `new reporter` command generates reporter boilerplate:

1. **Creates** reporter class file
2. **Generates** test file (optional)
3. **Creates** documentation (optional)
4. **Configures** output format settings

## Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `basic` | Minimal reporter structure | Simple outputs |
| `full` | All features (filtering, sorting) | Production reporters |

## Examples

### Basic Reporter

```bash
truthound new reporter my_reporter
```

Generated structure:
```
./
├── my_reporter.py
└── tests/
    └── test_my_reporter.py
```

Generated code (`my_reporter.py`):
```python
from truthound.reporters.base import Reporter
from truthound.validators.report import ValidationReport

class MyReporter(Reporter):
    """Custom reporter: my_reporter"""

    name = "my_reporter"
    extension = ".txt"
    content_type = "text/plain"

    def render(self, report: ValidationReport) -> str:
        """Render the validation report."""
        lines = []
        lines.append(f"Validation Report")
        lines.append(f"=" * 40)
        lines.append(f"Total Issues: {len(report.issues)}")
        lines.append("")

        for issue in report.issues:
            lines.append(f"[{issue.severity}] {issue.validator}: {issue.message}")

        return "\n".join(lines)
```

### JSON Reporter

```bash
truthound new reporter json_export --extension .json --content-type application/json
```

Generated code:
```python
import json
from truthound.reporters.base import Reporter
from truthound.validators.report import ValidationReport

class JsonExportReporter(Reporter):
    """Custom reporter: json_export"""

    name = "json_export"
    extension = ".json"
    content_type = "application/json"

    def render(self, report: ValidationReport) -> str:
        """Render the validation report as JSON."""
        data = {
            "summary": {
                "total_issues": len(report.issues),
                "by_severity": self._count_by_severity(report),
            },
            "issues": [
                {
                    "validator": issue.validator,
                    "column": issue.column,
                    "message": issue.message,
                    "severity": issue.severity,
                }
                for issue in report.issues
            ],
        }
        return json.dumps(data, indent=2)

    def _count_by_severity(self, report):
        counts = {}
        for issue in report.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
```

### HTML Reporter

```bash
truthound new reporter html_export --extension .html --content-type text/html
```

Generated code:
```python
from truthound.reporters.base import Reporter
from truthound.validators.report import ValidationReport

class HtmlExportReporter(Reporter):
    """Custom reporter: html_export"""

    name = "html_export"
    extension = ".html"
    content_type = "text/html"

    def render(self, report: ValidationReport) -> str:
        """Render the validation report as HTML."""
        html = ['<!DOCTYPE html>', '<html>', '<head>',
                '<title>Validation Report</title>',
                '<style>',
                'table { border-collapse: collapse; width: 100%; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                '.critical { background-color: #ffcccc; }',
                '.high { background-color: #ffe6cc; }',
                '.medium { background-color: #ffffcc; }',
                '.low { background-color: #e6ffcc; }',
                '</style>',
                '</head>', '<body>',
                f'<h1>Validation Report</h1>',
                f'<p>Total Issues: {len(report.issues)}</p>',
                '<table>',
                '<tr><th>Severity</th><th>Validator</th><th>Column</th><th>Message</th></tr>']

        for issue in report.issues:
            html.append(
                f'<tr class="{issue.severity.lower()}">'
                f'<td>{issue.severity}</td>'
                f'<td>{issue.validator}</td>'
                f'<td>{issue.column or "-"}</td>'
                f'<td>{issue.message}</td>'
                f'</tr>'
            )

        html.extend(['</table>', '</body>', '</html>'])
        return '\n'.join(html)
```

### Full Featured Reporter

```bash
truthound new reporter detailed_report \
  --template full \
  --docs \
  --author "Data Team" \
  --description "Detailed validation report with filtering and sorting"
```

Generated structure:
```
./
├── detailed_report.py
├── README.md
└── tests/
    └── test_detailed_report.py
```

Generated code (`detailed_report.py` with full template):
```python
from typing import Optional, List
from truthound.reporters.base import Reporter
from truthound.validators.report import ValidationReport, ValidationIssue

class DetailedReportReporter(Reporter):
    """Detailed validation report with filtering and sorting"""

    name = "detailed_report"
    extension = ".txt"
    content_type = "text/plain"

    def __init__(
        self,
        min_severity: Optional[str] = None,
        sort_by: str = "severity",
        include_validators: Optional[List[str]] = None,
        exclude_validators: Optional[List[str]] = None,
    ):
        self.min_severity = min_severity
        self.sort_by = sort_by
        self.include_validators = include_validators
        self.exclude_validators = exclude_validators

    def render(self, report: ValidationReport) -> str:
        """Render the validation report with filtering and sorting."""
        issues = self._filter_issues(report.issues)
        issues = self._sort_issues(issues)
        return self._format_output(issues)

    def _filter_issues(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        # Filtering logic
        ...

    def _sort_issues(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        # Sorting logic
        ...

    def _format_output(self, issues: List[ValidationIssue]) -> str:
        # Formatting logic
        ...
```

### XML Reporter

```bash
truthound new reporter xml_export \
  --extension .xml \
  --content-type application/xml \
  --description "XML format validation report"
```

### Custom Output Directory

```bash
truthound new reporter my_reporter -o ./reporters/
```

Creates files in `./reporters/` directory.

## Template Comparison

| Feature | basic | full |
|---------|-------|------|
| Basic rendering | Yes | Yes |
| Filtering | - | Yes |
| Sorting | - | Yes |
| Configuration | - | Yes |
| Tests | Yes | Yes |
| Documentation | Optional | Optional |

## Common Content Types

| Format | Extension | Content-Type |
|--------|-----------|--------------|
| Plain Text | `.txt` | `text/plain` |
| JSON | `.json` | `application/json` |
| HTML | `.html` | `text/html` |
| XML | `.xml` | `application/xml` |
| CSV | `.csv` | `text/csv` |
| Markdown | `.md` | `text/markdown` |

## Use Cases

### 1. Custom JSON Output

```bash
truthound new reporter api_response \
  --extension .json \
  --content-type application/json \
  --description "API-friendly JSON response format"
```

### 2. Slack-Friendly Format

```bash
truthound new reporter slack_message \
  --template full \
  --extension .txt \
  --description "Slack message format with emoji indicators"
```

### 3. CSV Export

```bash
truthound new reporter csv_export \
  --extension .csv \
  --content-type text/csv \
  --description "CSV export for spreadsheet analysis"
```

### 4. Custom HTML Dashboard

```bash
truthound new reporter dashboard \
  --template full \
  --extension .html \
  --content-type text/html \
  --docs
```

## Generated Test Code

```python
import pytest
from my_reporter import MyReporter
from truthound.validators.report import ValidationReport, ValidationIssue

class TestMyReporter:
    def test_render_empty_report(self):
        reporter = MyReporter()
        report = ValidationReport(issues=[])
        result = reporter.render(report)
        assert "Total Issues: 0" in result

    def test_render_with_issues(self):
        reporter = MyReporter()
        report = ValidationReport(issues=[
            ValidationIssue(
                validator="test",
                column="col1",
                message="Test issue",
                severity="MEDIUM",
            )
        ])
        result = reporter.render(report)
        assert "Test issue" in result
        assert "MEDIUM" in result
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Generation error |
| 2 | Invalid arguments |

## Related Commands

- [`new validator`](new-validator.md) - Create custom validator
- [`new plugin`](new-plugin.md) - Create plugin package
- [`new templates`](new-templates.md) - List available templates

## See Also

- [Scaffolding Overview](index.md)
- [Reporters Guide](../../guides/reporters.md)
