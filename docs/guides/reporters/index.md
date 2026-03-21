# Reporters Guide

Truthound reporters render a `ValidationRunResult` into formats that work for
humans, CI systems, and downstream automation. In 3.0 the canonical runtime
input is the immutable validation kernel result returned by `th.check()`.

Use this guide when you need to:

- render the same validation run in multiple formats
- publish results into CI annotations or summaries
- export structured output for automation
- build a custom reporter on top of the 3.0 reporter SDK

## Quick Start

```python
import truthound as th
from truthound.reporters import get_reporter

run_result = th.check("data.csv")

# Built-in factory reporters
json_output = get_reporter("json").render(run_result)
markdown_output = get_reporter("markdown").render(run_result)

# Write directly to disk
get_reporter("html").write(run_result, "reports/data-quality.html")
```

## Choose A Path

### Use built-in factory reporters

Use `get_reporter(...)` when you want the standard Truthound formats:

- `console`
- `json`
- `markdown`
- `html`
- `ci`
- `github`
- `gitlab`
- `jenkins`
- `azure`
- `circleci`
- `bitbucket`

### Use SDK template reporters

Use the SDK templates when you need extra export formats such as:

- `YAMLReporter`
- `CSVReporter`
- `NDJSONReporter`
- `JUnitXMLReporter`
- `TableReporter`

These live under `truthound.reporters.sdk.templates` and are not part of the
thin default factory surface.

### Build a custom reporter

Use `ValidationReporter` or the reporter SDK when you want:

- custom formatting rules
- extra filtering/grouping
- organization-specific exports
- a reporter plugin that can be reused across projects

## Input Model

`ValidationRunResult` is the canonical reporter input in Truthound 3.0.
Reporters can still adapt legacy compatibility objects internally, but current
docs and new custom reporters should target the run result directly.

```python
import truthound as th

run_result = th.check("data.csv")

print(run_result.success)
print(len(run_result.checks))
print(len(run_result.issues))
print(run_result.source)
print(run_result.run_id)
```

Common issue fields in 3.0:

- `issue.validator_name`
- `issue.issue_type`
- `issue.column`
- `issue.severity`
- `issue.message`
- `issue.count`

## Built-in Reporter Registry

### Standard output

```python
from truthound.reporters import get_reporter

reporter = get_reporter("console")
print(reporter.render(run_result))
```

### Multiple outputs from one run

```python
from truthound.reporters import get_reporter

for fmt in ["json", "markdown", "html"]:
    reporter = get_reporter(fmt)
    reporter.write(run_result, f"reports/result{reporter.file_extension}")
```

### CI-native output

```python
from truthound.reporters import get_reporter

github_reporter = get_reporter("github")
exit_code = github_reporter.report_to_ci(run_result)
```

## SDK Template Reporters

These are prebuilt reporter classes exposed through the SDK:

```python
from truthound.reporters.sdk.templates import (
    CSVReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,
    YAMLReporter,
)

yaml_output = YAMLReporter().render(run_result)
xml_output = JUnitXMLReporter().render(run_result)
```

Recommended uses:

| Reporter | Best for |
|----------|----------|
| `YAMLReporter` | Human-readable structured exports |
| `CSVReporter` | Spreadsheet workflows |
| `NDJSONReporter` | Event/log pipelines |
| `JUnitXMLReporter` | Test-report ingestion |
| `TableReporter` | Compact text tables |

## Common Workflows

### JSON for automation, Markdown for humans

```python
from truthound.reporters import get_reporter

json_reporter = get_reporter("json")
markdown_reporter = get_reporter("markdown")

json_reporter.write(run_result, "artifacts/validation.json")
markdown_reporter.write(run_result, "artifacts/validation.md")
```

### HTML report with graceful dependency expectations

```python
from truthound.reporters import get_reporter

html_reporter = get_reporter("html", title="Customer Quality Report")
html_reporter.write(run_result, "site/customer-quality.html")
```

`html` output uses `jinja2`. Install the reporting extras if you want the full
HTML renderer available in every environment.

### Custom reporter registration

```python
from truthound.reporters.base import ReporterConfig, ValidationReporter
from truthound.reporters.factory import register_reporter


@register_reporter("severity-summary")
class SeveritySummaryReporter(ValidationReporter[ReporterConfig]):
    name = "severity-summary"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, data) -> str:
        presentation = self.present(data)
        counts = presentation.issue_counts_by_severity
        return "\n".join(
            [
                f"status={presentation.status}",
                f"critical={counts.get('critical', 0)}",
                f"high={counts.get('high', 0)}",
                f"medium={counts.get('medium', 0)}",
                f"low={counts.get('low', 0)}",
            ]
        )
```

## Reporter Families

| Family | Primary contract | Start here |
|--------|-------------------|------------|
| Console | terminal-first human output | [Console Reporter](console.md) |
| JSON/YAML | machine-readable structured output | [JSON & YAML](json-yaml.md) |
| HTML/Markdown | docs, artifacts, static publishing | [HTML & Markdown](html-markdown.md) |
| CI/CD | platform annotations and summaries | [CI/CD Reporters](ci-reporters.md) |
| SDK | custom reporter construction | [Reporter SDK](custom-sdk.md) |

## Troubleshooting

### I need YAML or JUnit from Python, but `get_reporter(...)` does not expose it

Use the SDK templates directly:

```python
from truthound.reporters.sdk.templates import YAMLReporter, JUnitXMLReporter
```

### I have a legacy report-like object

Current code should prefer passing `ValidationRunResult`. Reporter adapters still
exist for compatibility, but new docs and new integrations should standardize on
the run result model.

### I want richer access than `run_result.issues`

Inside custom reporters, use:

```python
presentation = self.present(run_result)
legacy_view = presentation.to_legacy_view()
```

`presentation` gives the shared 3.0 rendering model, while `legacy_view`
provides a compatibility projection for helper mixins that operate on flattened
result rows.

## See Also

- [Python API: Reporters](../../python-api/reporters.md)
- [Console Reporter](console.md)
- [JSON & YAML](json-yaml.md)
- [HTML & Markdown](html-markdown.md)
- [CI/CD Reporters](ci-reporters.md)
- [Reporter SDK](custom-sdk.md)
