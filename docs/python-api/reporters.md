# Reporters

Output surfaces for `ValidationRunResult`.

## Canonical Contract

Truthound 3.0 reporter APIs are built around `ValidationRunResult`.

- `th.check()` returns `ValidationRunResult`
- `truthound.reporters` canonicalizes supported inputs into that model
- built-in validation reporters render from a shared `RunPresentation`
- persisted `truthound.stores.results.ValidationResult` objects are still accepted through an adapter boundary, but they are not the preferred runtime contract

If you are starting new code, pass `ValidationRunResult` directly.

## Choose The Right Reporter Family

Reporter surfaces in Truthound are related, but they are not all the same
registry or subsystem.

| Family | Entry point | Primary input | What it is for |
|--------|-------------|---------------|----------------|
| Built-in validation reporters | `truthound.reporters.get_reporter(...)` | `ValidationRunResult` | JSON, Markdown, HTML, console, and CI aliases |
| CI-native reporters | `truthound.reporters.ci` | `ValidationRunResult` | Platform-specific annotations, summaries, and artifacts |
| SDK templates | `truthound.reporters.sdk.templates` | `ValidationRunResult` | YAML, CSV, NDJSON, JUnit XML, table output |
| Quality reporting subsystem | `truthound.reporters.quality` | quality score/reportable objects | Rule quality score reports, filters, and pipelines |

`truthound.reporters.quality` is not part of the built-in `get_reporter(...)` registry.

## Quick Start

```python
import truthound as th
from truthound.reporters import get_reporter
from truthound.reporters.sdk.templates import YAMLReporter

run_result = th.check("data.csv")

json_output = get_reporter("json").render(run_result)
markdown_output = get_reporter("markdown").render(run_result)
yaml_output = YAMLReporter().render(run_result)

get_reporter("html").write(run_result, "report.html")
```

## Built-in `get_reporter(...)` Registry

The built-in registry is intentionally thin. It covers the standard validation
run formats and CI aliases, not every reporter-related subsystem in the repo.

| Format | Reporter | Notes |
|--------|----------|-------|
| `console` | `ConsoleReporter` | Rich terminal output |
| `json` | `JSONReporter` | Machine-readable validation payload |
| `markdown` | `MarkdownReporter` | Docs, PR comments, wiki output |
| `html` | `HTMLReporter` | Browser-friendly validation report pages |
| `ci` | platform auto-detect | Selects the active CI reporter |
| `github` | `GitHubActionsReporter` | GitHub annotations and summaries |
| `gitlab` | `GitLabCIReporter` | GitLab CI/code-quality output |
| `jenkins` | `JenkinsReporter` | Jenkins-oriented CI output |
| `azure` | `AzureDevOpsReporter` | Azure DevOps logging commands |
| `circleci` | `CircleCIReporter` | CircleCI metadata output |
| `bitbucket` | `BitbucketPipelinesReporter` | Bitbucket pipeline summaries |

Use direct imports for families that are outside this registry:

- SDK templates: `truthound.reporters.sdk.templates`
- custom reporter authoring: `truthound.reporters.sdk`
- quality subsystem: `truthound.reporters.quality`

## Shared Input Model

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
the run result. That is what keeps JSON, HTML, Markdown, and CI output aligned.

## Built-in Validation Reporters

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

## CI-native Reporters

The CI family lives under `truthound.reporters.ci`.

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter()
exit_code = reporter.report_to_ci(run_result)
```

The public input remains `ValidationRunResult`, but CI reporters internally
format against a `LegacyValidationResultView` compatibility projection after
building the shared presentation model. That split is intentional: external code
passes the canonical run result, while CI emitters get the flattened shape they
need for provider-specific annotations and summaries.

## SDK Templates And Custom Reporters

Truthound also ships optional template reporters under
`truthound.reporters.sdk.templates`. These are not part of the built-in
`get_reporter(...)` registry, but they are available for direct import:

| Class | Import path | Typical use |
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

For new custom reporters, subclass `ValidationReporter` so you inherit the 3.0
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

Register it at runtime if you want it available through the built-in registry:

```python
from truthound.reporters import register_reporter

register_reporter("csv", CSVReporter)
```

## Related But Separate: Quality Reporting

`truthound.reporters.quality` is a separate quality-score reporting subsystem.
It uses `get_quality_reporter(...)` and quality score/reportable objects, not
`ValidationRunResult`, as its primary input.

```python
from truthound.reporters.quality import get_quality_reporter

reporter = get_quality_reporter("console")
```

Use it when you are reporting rule-quality scores from the profiler, not when
you are rendering a validation run returned by `th.check()`.

## Compatibility Notes

- Passing `truthound.stores.results.ValidationResult` directly to a reporter is a compatibility path, not the preferred 3.0 contract.
- If you are migrating old docs or plugins, treat `ValidationRunResult` as the source of truth and adapt older DTOs at the boundary.
- Legacy `Report` objects are not the canonical reporter input in 3.0.
- For upgrade guidance on removed runtime `Report` assumptions, see [Migration to 3.0](../guides/migration-3.0.md).

## Related Reading

- [Core Functions](core-functions.md)
- [Guides: Reporters](../guides/reporters/index.md)
- [Guides: Reporter SDK](../guides/reporter-sdk.md)
- [Guides: Quality Reporter](../guides/quality-reporter.md)
- [Guides: DataDocs](../guides/datadocs/index.md)
- [Migration to 3.0](../guides/migration-3.0.md)
