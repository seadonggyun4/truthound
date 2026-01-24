# Reporters Guide

This guide covers output formatting with Truthound's Python API. It includes practical workflows for generating reports in various formats and creating custom reporters.

---

## Quick Start

```python
from truthound.reporters import get_reporter
import truthound as th

# Run validation
report = th.check("data.csv")

# Output as JSON
json_reporter = get_reporter("json")
json_output = json_reporter.render(report)

# Write to file
json_reporter.write(report, "results.json")
```

---

## Common Workflows

### Workflow 1: Multiple Output Formats

```python
from truthound.reporters import get_reporter
import truthound as th

report = th.check("data.csv")

# Generate multiple formats
formats = ["json", "markdown", "html"]
for fmt in formats:
    reporter = get_reporter(fmt)
    reporter.write(report, f"results.{reporter.file_extension}")
```

### Workflow 2: CI/CD Integration (GitHub Actions)

```python
from truthound.reporters import get_reporter
import truthound as th

report = th.check("data.csv")

# GitHub Actions reporter with annotations
gh_reporter = get_reporter("github")
gh_reporter.write(report, "results.json")

# Also output JUnit XML for test reporting
junit_reporter = get_reporter("junit")
junit_reporter.write(report, "results.xml")
```

### Workflow 3: Custom Reporter Implementation

```python
from truthound.reporters import BaseReporter, register_reporter

class CSVReporter(BaseReporter):
    name = "csv"
    file_extension = "csv"
    content_type = "text/csv"

    def render(self, report):
        lines = ["column,issue_type,severity,count"]
        for issue in report.issues:
            lines.append(
                f"{issue.column},{issue.issue_type},{issue.severity},{issue.count}"
            )
        return "\n".join(lines)

# Register custom reporter
register_reporter("csv", CSVReporter)

# Use custom reporter
reporter = get_reporter("csv")
csv_output = reporter.render(report)
```

### Workflow 4: Styled Console Output

```python
from truthound.reporters import get_reporter

# Console reporter with configuration
reporter = get_reporter(
    "console",
    show_summary=True,
    show_issues=True,
    group_by="column",
    max_issues=50,
)

# Print to terminal with Rich formatting
output = reporter.render(report)
print(output)
```

---

## Full Documentation

The Truthound reporter system outputs validation results in various formats.

## Document Structure

| Document | Description |
|----------|-------------|
| [Console Reporter](console.md) | Terminal output (Rich-based) |
| [JSON & YAML](json-yaml.md) | JSON, YAML, NDJSON formats |
| [HTML & Markdown](html-markdown.md) | HTML, Markdown, Table formats |
| [CI/CD Reporters](ci-reporters.md) | GitHub Actions, GitLab CI, Jenkins, etc. |
| [Reporter SDK](custom-sdk.md) | Custom reporter development SDK |

---

## Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Reporter Factory                         │
│              get_reporter(format, **config)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    JSON     │    │   Console   │    │  Markdown   │
    │  Reporter   │    │  Reporter   │    │  Reporter   │
    └─────────────┘    └─────────────┘    └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │    HTML     │
                       │  Reporter   │
                       └─────────────┘
```

### Key Features

- **Multiple Formats**: JSON, Console, Markdown, HTML, YAML, CSV, JUnit XML
- **Unified Interface**: Same API across all reporters
- **Customization**: Title, theme, template configuration
- **Extensibility**: Register custom reporters at runtime

---

## Quick Start

### Basic Usage

```python
from truthound.reporters import get_reporter
import truthound as th

# Run validation
result = th.check("data.csv")

# Create reporter
reporter = get_reporter("json")

# Render as string
json_output = reporter.render(result)

# Write to file
reporter.write(result, "report.json")
```

### Available Formats

| Format | Dependencies | Use Case |
|--------|--------------|----------|
| `json` | (built-in) | API integration, programmatic access |
| `console` | rich | Terminal output, debugging |
| `markdown` | (built-in) | Documentation, GitHub/GitLab |
| `html` | jinja2 | Web dashboards, email reports |

### Format Availability Check

```python
from truthound.reporters.factory import list_available_formats, is_format_available

# List all available formats
print(list_available_formats())
# ['console', 'json', 'markdown', 'html']

# Check specific format
if is_format_available("html"):
    reporter = get_reporter("html")
```

---

## Reporter Comparison

### Output Format Comparison

| Reporter | Output Format | File Extension | Primary Use |
|----------|---------------|----------------|-------------|
| ConsoleReporter | Text (Rich) | - | Terminal output |
| JSONReporter | JSON | `.json` | API, automation |
| YAMLReporter | YAML | `.yaml` | Configuration, readability |
| MarkdownReporter | Markdown | `.md` | Documentation |
| HTMLReporter | HTML | `.html` | Web reports |
| TableReporter | ASCII/Markdown | `.txt` | Simple tables |
| JUnitXMLReporter | XML | `.xml` | CI/CD integration |

### CI/CD Platform Comparison

| Platform | Reporter | Key Features |
|----------|----------|--------------|
| GitHub Actions | `GitHubActionsReporter` | Annotations, Step Summaries |
| GitLab CI | `GitLabCIReporter` | Code Quality JSON, JUnit XML |
| Jenkins | `JenkinsReporter` | JUnit XML, warnings-ng JSON |
| Azure DevOps | `AzureDevOpsReporter` | VSO Commands, Variables |
| CircleCI | `CircleCIReporter` | Test Metadata, Artifacts |
| Bitbucket Pipelines | `BitbucketPipelinesReporter` | Reports, Annotations |

### SDK Template Comparison

| Template | Use Case | Features |
|----------|----------|----------|
| CSVReporter | Data export | Spreadsheet compatible |
| YAMLReporter | Configuration files | Human readable |
| JUnitXMLReporter | CI/CD | Test framework compatible |
| NDJSONReporter | Log collection | ELK, Splunk integration |
| TableReporter | Terminal | 4 style options |

---

## Common Interface

All reporters implement the same interface:

```python
class BaseReporter(Generic[ConfigT, InputT], ABC):
    name: str                   # Reporter name
    file_extension: str         # Default file extension
    content_type: str           # MIME type

    def render(self, data: InputT) -> str:
        """Render result as string."""
        ...

    def write(self, data: InputT, path: str | Path | None = None) -> Path:
        """Write result to file. Returns written path."""
        ...

    def report(self, data: InputT, path: str | Path | None = None) -> str:
        """Render and optionally write to file. Returns rendered string."""
        ...
```

> **Note**: `ConsoleReporter` has an additional `print(data)` method for direct terminal output.

---

## Custom Reporter Registration

### Using Decorator

```python
from truthound.reporters import register_reporter
from truthound.reporters.base import ValidationReporter, ReporterConfig

@register_reporter("my_custom")
class MyCustomReporter(ValidationReporter[ReporterConfig]):
    name = "my_custom"
    file_extension = ".custom"

    def render(self, data):
        return f"Custom: {data.status.value}"
```

### Using SDK Decorator

```python
from truthound.reporters.sdk import create_reporter

@create_reporter("simple", extension=".txt")
def render_simple(result, config):
    return f"Status: {result.status.value}"
```

For details, see the [Reporter SDK](custom-sdk.md) documentation.

---

## Integration Examples

### GitHub Actions

```yaml
name: Data Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install truthound

      - name: Run validation
        run: |
          python -c "
          import truthound as th
          from truthound.reporters.ci import GitHubActionsReporter

          result = th.check('data/*.csv')
          reporter = GitHubActionsReporter()
          exit_code = reporter.report_to_ci(result)
          exit(exit_code)
          "
```

### FastAPI Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from truthound.reporters import get_reporter
import truthound as th

app = FastAPI()

@app.post("/validate")
async def validate_data(file_path: str, format: str = "json"):
    result = th.check(file_path)
    reporter = get_reporter(format)
    content = reporter.render(result)

    if format == "html":
        return HTMLResponse(content=content)
    return JSONResponse(content=json.loads(content))
```

### Airflow Task

```python
from airflow.decorators import task
from truthound.reporters import get_reporter

@task
def validate_and_report(data_path: str, report_path: str):
    import truthound as th

    result = th.check(data_path)

    # Generate HTML report
    reporter = get_reporter("html", title="Daily Data Quality")
    reporter.write(result, report_path)

    # Fail task on validation failure
    if not result.success:
        raise ValueError(f"Data quality check failed for {data_path}")

    return report_path
```

---

## Summary

Truthound reporters provide flexible output formatting:

- **4 built-in formats**: JSON, Console, Markdown, HTML
- **6 CI/CD reporters**: GitHub Actions, GitLab CI, Jenkins, Azure DevOps, CircleCI, Bitbucket
- **5 SDK templates**: CSV, YAML, JUnit XML, NDJSON, Table
- **6 SDK Mixins**: Formatting, Aggregation, Filtering, Serialization, Templating, Streaming
- **Schema validation**: JSON, XML, CSV schemas and validation utilities
- **Testing utilities**: Mock data, assertions, benchmarking
- **Unified interface**: Same API across all reporters
- **Extensibility**: Register custom reporters with `@register_reporter`

For details, see each document:

- [Console Reporter](console.md) - Terminal output
- [JSON & YAML](json-yaml.md) - Structured data formats
- [HTML & Markdown](html-markdown.md) - Document and web reports
- [CI/CD Reporters](ci-reporters.md) - CI/CD platform integration
- [Reporter SDK](custom-sdk.md) - Custom reporter development
