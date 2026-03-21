# Reporters

Output formatters for validation runs.

## Canonical Contract

Truthound 3.0 reporters are built around `ValidationRunResult`.

- `th.check()` returns `ValidationRunResult`
- `truthound.reporters` canonicalizes supported inputs into that model
- persisted `truthound.stores.results.ValidationResult` objects are still accepted through a compatibility adapter, but they are no longer the preferred runtime contract
- legacy `Report` objects are not the canonical reporter input in 3.0

If you are starting new code, pass `ValidationRunResult` directly.

## Quick Start

```python
import truthound as th
from truthound.reporters import get_reporter

run_result = th.check("data.csv")

json_output = get_reporter("json").render(run_result)
markdown_output = get_reporter("markdown").render(run_result)

get_reporter("html").write(run_result, "report.html")
```

## Built-in Reporter Registry

The default reporter factory currently exposes these formats:

| Format | Reporter | Notes |
|--------|----------|-------|
| `console` | `ConsoleReporter` | Rich terminal output |
| `json` | `JSONReporter` | Machine-readable API/export format |
| `markdown` | `MarkdownReporter` | Docs, PR comments, wiki output |
| `html` | `HTMLReporter` | Browser-friendly report pages |
| `ci` | platform auto-detect | Selects the active CI reporter |
| `github` | `GitHubActionsReporter` | GitHub annotations and summaries |
| `gitlab` | `GitLabCIReporter` | GitLab CI/code-quality output |
| `jenkins` | `JenkinsReporter` | Jenkins-friendly report output |
| `azure` | `AzureDevOpsReporter` | Azure DevOps logging commands |
| `circleci` | `CircleCIReporter` | CircleCI metadata output |
| `bitbucket` | `BitbucketPipelinesReporter` | Bitbucket pipeline summaries |

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json", indent=2)
payload = reporter.render(run_result)

ci_reporter = get_reporter("github")
ci_payload = ci_reporter.render(run_result)
```

## Reporter Input Model

Use `ValidationRunResult` terminology throughout your code and docs:

```python
import truthound as th
from truthound.reporters import get_reporter

run_result = th.check("data.csv")

print(run_result.run_id)
print(len(run_result.checks))
print(len(run_result.issues))

reporter = get_reporter("json")
print(reporter.render(run_result))
```

Under the hood, validation reporters build a shared `RunPresentation` model from
the run result, which is what keeps JSON, HTML, Markdown, and CI outputs
consistent.

## Core Reporter Classes

### `ConsoleReporter`

```python
from truthound.reporters.console_reporter import ConsoleReporter

reporter = ConsoleReporter(color=True, compact=False)
print(reporter.render(run_result))
```

### `JSONReporter`

```python
from truthound.reporters.json_reporter import JSONReporter

reporter = JSONReporter(indent=2, include_null_values=True)
print(reporter.render(run_result))
```

Representative output shape:

```json
{
  "report_type": "validation",
  "generated_at": "2026-03-21 20:41:09",
  "title": "Truthound Validation Report",
  "result": {
    "run_id": "run_20260321_204109_2770ac12",
    "run_time": "2026-03-21T20:41:09.672591",
    "data_asset": "dict",
    "status": "failure",
    "success": false
  },
  "statistics": {
    "total_validators": 5,
    "passed_validators": 3,
    "failed_validators": 2
  },
  "issues": [
    {
      "validator": "unique",
      "column": "id",
      "issue_type": "unique_violation",
      "severity": "critical"
    }
  ]
}
```

### `MarkdownReporter`

```python
from truthound.reporters.markdown_reporter import MarkdownReporter

reporter = MarkdownReporter(include_toc=True, include_badges=True)
markdown = reporter.render(run_result)
```

### `HTMLReporter`

```python
from truthound.reporters.html_reporter import HTMLReporter

reporter = HTMLReporter(title="Customer Data Quality")
reporter.write(run_result, "report.html")
```

`HTMLReporter` requires `jinja2`. Install the reporting extras if you want HTML
output in production docs or CI.

## CI Reporters

The CI reporters live under `truthound.reporters.ci`.

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter()
payload = reporter.render(run_result)
exit_code = reporter.report_to_ci(run_result)
```

Use these when you need native CI annotations, job summaries, or code-quality
artifacts instead of generic JSON/Markdown files.

## SDK Template Reporters

Truthound also ships optional reporter templates under
`truthound.reporters.sdk.templates`. These are not the default registry formats
returned by `get_reporter(...)`, but they are available for direct import:

| Class | Import Path | Typical Use |
|------|-------------|-------------|
| `YAMLReporter` | `truthound.reporters.sdk.templates` | Human-readable structured export |
| `CSVReporter` | `truthound.reporters.sdk.templates` | Spreadsheet workflows |
| `NDJSONReporter` | `truthound.reporters.sdk.templates` | Log/event pipelines |
| `JUnitXMLReporter` | `truthound.reporters.sdk.templates` | Test-report integration |
| `TableReporter` | `truthound.reporters.sdk.templates` | Plain text or compact tables |

```python
from truthound.reporters.sdk.templates import JUnitXMLReporter, YAMLReporter

yaml_output = YAMLReporter().render(run_result)
xml_output = JUnitXMLReporter().render(run_result)
```

## Custom Reporters

For new reporters, subclass `ValidationReporter` so you inherit the 3.0
canonicalization and presentation flow.

```python
from truthound.core.results import ValidationRunResult
from truthound.reporters.base import ReporterConfig, ValidationReporter


class CSVReporter(ValidationReporter[ReporterConfig]):
    name = "csv"
    file_extension = ".csv"
    content_type = "text/csv"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, data: ValidationRunResult) -> str:
        presentation = self.present(data)
        lines = ["validator,column,issue_type,severity,count"]
        for issue in presentation.issues:
            lines.append(
                f"{issue.validator_name},{issue.column},{issue.issue_type},{issue.severity},{issue.count}"
            )
        return "\n".join(lines)
```

Register it at runtime if you want it available through the factory:

```python
from truthound.reporters import register_reporter

register_reporter("csv", CSVReporter)
```

## Compatibility Notes

- Passing `truthound.stores.results.ValidationResult` directly to a reporter is a compatibility path, not the preferred 3.0 contract.
- If you are migrating old docs or plugins, treat `ValidationRunResult` as the source of truth and adapt older DTOs at the boundary.
- For upgrade guidance on removed runtime `Report` assumptions, see [Migration to 3.0](../guides/migration-3.0.md).

## Related Reading

- [Core Functions](core-functions.md)
- [Validators](validators.md)
- [Guides: Reporters](../guides/reporters/index.md)
- [Guides: DataDocs](../guides/datadocs/index.md)
- [Migration to 3.0](../guides/migration-3.0.md)
