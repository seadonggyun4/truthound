# Reporters

Output formatters for validation results.

## Overview

Truthound provides multiple output formats for validation reports:

| Reporter | Format | Use Case | Module |
|----------|--------|----------|--------|
| `ConsoleReporter` | Terminal output | Interactive development | `truthound.reporters` |
| `JSONReporter` | JSON | API integration | `truthound.reporters` |
| `HTMLReporter` | HTML | Dashboards, sharing | `truthound.reporters` |
| `MarkdownReporter` | Markdown | Documentation | `truthound.reporters` |
| `JUnitXMLReporter` | JUnit XML | CI/CD integration | `truthound.reporters.sdk.templates` |
| `YAMLReporter` | YAML | Human-readable structured data | `truthound.reporters.sdk.templates` |
| `CSVReporter` | CSV | Spreadsheets | `truthound.reporters.sdk.templates` |
| `NDJSONReporter` | NDJSON | Streaming/log aggregation | `truthound.reporters.sdk.templates` |
| `TableReporter` | ASCII/Unicode table | Terminal display | `truthound.reporters.sdk.templates` |

!!! important "Report vs ValidationResult"
    **Reporters accept `ValidationResult`** from `truthound.stores.results`, not `Report` from `truthound.report`.

    - **`Report`**: Returned by `th.check()` - simple report with `issues: list[ValidationIssue]`, `has_issues`, `has_critical`, `print()`, `to_json()`
    - **`ValidationResult`**: Storage-oriented result with `run_id`, `run_time`, `results: list[ValidatorResult]`, `statistics`, `tags`

    For quick output with `th.check()`, use the built-in methods:

    ```python
    import truthound as th

    report = th.check("data.csv")
    report.print()           # Console output
    print(report.to_json())  # JSON output
    ```

    For full reporter functionality, create a `ValidationResult`:

    ```python
    from datetime import datetime
    from truthound.stores.results import (
        ValidationResult,
        ValidatorResult,
        ResultStatus,
        ResultStatistics,
    )

    # Create ValidationResult from Report
    result = ValidationResult(
        run_id="run-001",
        run_time=datetime.now(),
        data_asset="data.csv",
        status=ResultStatus.FAILURE if report.has_issues else ResultStatus.SUCCESS,
        results=[ValidatorResult.from_issue(i) for i in report.issues],
        statistics=ResultStatistics(
            total_issues=len(report.issues),
            total_rows=report.row_count,
            total_columns=report.column_count,
        ),
    )
    ```

## BaseReporter

All reporters inherit from `BaseReporter`.

### Definition

```python
from truthound.reporters.base import BaseReporter, ReporterConfig

class BaseReporter(ABC, Generic[ConfigT, InputT]):
    """Abstract base class for all reporters."""

    @abstractmethod
    def render(self, input_data: InputT) -> str:
        """Render input data to string format."""

    def write(self, input_data: InputT, output: Path | str | IO) -> None:
        """Write rendered output to file or stream."""

    def __call__(self, input_data: InputT) -> str:
        """Shorthand for render()."""
        return self.render(input_data)
```

### ReporterConfig

```python
from truthound.reporters.base import ReporterConfig

@dataclass
class ReporterConfig:
    """Base configuration for reporters."""

    output_path: str | Path | None = None
    title: str = "Truthound Validation Report"
    include_metadata: bool = True
    include_statistics: bool = True
    include_details: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    max_sample_values: int = 5
```

---

## Built-in Reporters

### ConsoleReporter

Rich terminal output with colors.

```python
from truthound.reporters.console_reporter import ConsoleReporter, ConsoleReporterConfig

# Basic usage (with ValidationResult)
reporter = ConsoleReporter()
print(reporter.render(result))

# With configuration
config = ConsoleReporterConfig(
    color=True,
    width=100,
    show_header=True,
    show_summary=True,
    show_issues_table=True,
    compact=False,
)
reporter = ConsoleReporter(config)
output = reporter.render(result)
```

### JSONReporter

JSON output for API integration.

```python
from truthound.reporters.json_reporter import JSONReporter, JSONReporterConfig

# Basic usage (with ValidationResult)
reporter = JSONReporter()
json_output = reporter.render(result)

# Pretty printed
config = JSONReporterConfig(
    indent=2,
    sort_keys=False,
    ensure_ascii=False,
    include_null_values=True,
    date_format="iso",  # "iso" or "timestamp"
)
reporter = JSONReporter(config)
reporter.write(result, "report.json")
```

### HTMLReporter

HTML reports for dashboards. Requires `jinja2` package (`pip install truthound[all]`).

```python
from truthound.reporters.html_reporter import HTMLReporter, HTMLReporterConfig

# Basic usage (with ValidationResult)
reporter = HTMLReporter()
html = reporter.render(result)

# With configuration
config = HTMLReporterConfig(
    title="Data Quality Report",
    template_path=None,        # Path to custom Jinja2 template
    inline_css=True,
    theme="light",             # "light" or "dark"
    include_charts=False,      # Future feature
    custom_css="",             # Additional CSS
)
reporter = HTMLReporter(config)
reporter.write(result, "report.html")
```

### JUnitXMLReporter

JUnit XML for CI/CD pipelines.

```python
from truthound.reporters.sdk.templates import JUnitXMLReporter, JUnitXMLReporterConfig

# Basic usage (with ValidationResult)
reporter = JUnitXMLReporter()
xml = reporter.render(result)

# With configuration
config = JUnitXMLReporterConfig(
    suite_name="DataQuality",
    include_passed=True,
    include_properties=True,
    include_system_out=True,
)
reporter = JUnitXMLReporter(config)
reporter.write(result, "junit-report.xml")
```

### MarkdownReporter

Markdown for documentation.

```python
from truthound.reporters.markdown_reporter import MarkdownReporter, MarkdownReporterConfig

# Basic usage (with ValidationResult)
reporter = MarkdownReporter()
md = reporter.render(result)

# With configuration
config = MarkdownReporterConfig(
    include_toc=True,
    heading_level=1,
    include_badges=True,
    table_style="github",  # "github" or "simple"
)
reporter = MarkdownReporter(config)
reporter.write(result, "report.md")
```

### YAMLReporter

YAML output format (requires PyYAML).

```python
from truthound.reporters.sdk.templates import YAMLReporter, YAMLReporterConfig

# Basic usage
reporter = YAMLReporter()
yaml_output = reporter.render(result)

# With configuration
config = YAMLReporterConfig(
    default_flow_style=False,
    indent=2,
    include_passed=False,
    sort_keys=False,
)
reporter = YAMLReporter(config)
reporter.write(result, "report.yaml")
```

### CSVReporter

CSV for spreadsheet analysis.

```python
from truthound.reporters.sdk.templates import CSVReporter, CSVReporterConfig

config = CSVReporterConfig(
    delimiter=",",
    quote_char='"',
    include_header=True,
    include_passed=False,
)
reporter = CSVReporter(config)
reporter.write(result, "issues.csv")
```

---

## CI Platform Reporters

Specialized reporters for CI/CD platforms.

### GitHub Actions

```python
from truthound.reporters.ci.github import GitHubActionsReporter

reporter = GitHubActionsReporter()
# Outputs ::error:: and ::warning:: annotations
print(reporter.render(report))
```

### GitLab CI

```python
from truthound.reporters.ci.gitlab import GitLabCIReporter

reporter = GitLabCIReporter()
# Outputs section markers
print(reporter.render(report))
```

### Azure DevOps

```python
from truthound.reporters.ci.azure import AzureDevOpsReporter

reporter = AzureDevOpsReporter()
# Outputs ##vso[task.logissue] commands
print(reporter.render(report))
```

### Jenkins

```python
from truthound.reporters.ci.jenkins import JenkinsReporter

reporter = JenkinsReporter()
print(reporter.render(report))
```

---

## Reporter SDK

Create custom reporters using the SDK.

### Using Decorators

```python
from truthound.reporters.sdk import create_reporter

@create_reporter("my_custom", extension=".txt")
def render_my_format(result, config):
    """Simple custom reporter."""
    lines = ["My Custom Report", "=" * 40]
    for r in result.results:
        if not r.success:
            lines.append(f"- {r.column}: {r.issue_type}")
    return "\n".join(lines)

# Use like any other reporter
from truthound.reporters import get_reporter
reporter = get_reporter("my_custom")
output = reporter.render(result)
```

### Using ReporterBuilder

```python
from truthound.reporters.sdk.builder import ReporterBuilder

# Fluent builder pattern
MyReporter = (
    ReporterBuilder("my_reporter")
    .with_extension(".txt")
    .with_content_type("text/plain")
    .with_renderer(lambda result, config: f"Issues: {result.statistics.total_issues}")
    .build()
)

reporter = MyReporter()
print(reporter.render(result))
```

### Subclassing ValidationReporter

```python
from truthound.reporters.base import ValidationReporter, ReporterConfig

class MyCustomReporter(ValidationReporter[ReporterConfig]):
    """A fully custom reporter."""

    name = "my_custom"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, result) -> str:
        lines = [f"Total Issues: {result.statistics.total_issues}"]
        for r in result.results:
            if not r.success:
                lines.append(f"[{r.severity}] {r.column}: {r.issue_type}")
        return "\n".join(lines)
```

### Using Mixins

```python
from truthound.reporters.sdk import (
    FormattingMixin,
    FilteringMixin,
    AggregationMixin,
)
from truthound.reporters.base import ValidationReporter, ReporterConfig

class EnhancedReporter(
    FormattingMixin,
    FilteringMixin,
    AggregationMixin,
    ValidationReporter[ReporterConfig]
):
    """Reporter with formatting and filtering capabilities."""

    name = "enhanced"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> ReporterConfig:
        return ReporterConfig()

    def render(self, result) -> str:
        # Use mixin methods
        failed_results = self.filter_failed(result.results)
        by_severity = self.group_by_severity(failed_results)

        lines = [f"Summary: {result.statistics.total_issues} issues"]
        for severity, results in by_severity.items():
            lines.append(f"\n{severity.upper()} ({len(results)}):")
            for r in results:
                lines.append(f"  - {r.validator_name}: {r.column}")
        return "\n".join(lines)
```

---

## Reporter Factory

Get reporters by format name.

```python
from truthound.reporters import get_reporter, register_reporter

# Get by format
reporter = get_reporter("json")
reporter = get_reporter("html")
reporter = get_reporter("console")
reporter = get_reporter("markdown")

# CI platforms (auto-detection or specific)
reporter = get_reporter("ci")        # Auto-detect CI platform
reporter = get_reporter("github")    # GitHub Actions
reporter = get_reporter("gitlab")    # GitLab CI
reporter = get_reporter("azure")     # Azure DevOps
reporter = get_reporter("jenkins")   # Jenkins

# Register custom reporter using decorator
@register_reporter("my_format")
class MyCustomReporter(BaseReporter):
    ...

# Then use it
reporter = get_reporter("my_format")
```

---

## Usage Examples

### Quick Output with Report

For simple use cases, use the built-in `Report` methods:

```python
import truthound as th

report = th.check("data.csv")

# Console output
report.print()

# JSON output
print(report.to_json())
json_string = report.to_json(indent=2)
```

### Using Reporters with ValidationResult

For full reporter functionality:

```python
from datetime import datetime
import truthound as th
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatus,
    ResultStatistics,
)
from truthound.reporters.json_reporter import JSONReporter

# 1. Run validation
report = th.check("data.csv")

# 2. Create ValidationResult
result = ValidationResult(
    run_id=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    run_time=datetime.now(),
    data_asset="data.csv",
    status=ResultStatus.FAILURE if report.has_issues else ResultStatus.SUCCESS,
    results=[ValidatorResult.from_issue(i) for i in report.issues],
    statistics=ResultStatistics(
        total_issues=len(report.issues),
        total_rows=report.row_count,
        total_columns=report.column_count,
    ),
)

# 3. Use reporters
JSONReporter().write(result, "output/report.json")
```

### Multiple Formats

```python
from truthound.reporters.json_reporter import JSONReporter
from truthound.reporters.html_reporter import HTMLReporter
from truthound.reporters.sdk.templates import JUnitXMLReporter

# Assuming `result` is a ValidationResult
JSONReporter().write(result, "report.json")
HTMLReporter().write(result, "report.html")
JUnitXMLReporter().write(result, "junit.xml")
```

### Custom Styling

```python
from truthound.reporters.html_reporter import HTMLReporter, HTMLReporterConfig

config = HTMLReporterConfig(
    title="Production Data Quality",
    theme="dark",
    custom_css="""
        .severity-critical { background: #ff0000; }
        .summary-card { border-radius: 8px; }
    """,
)
reporter = HTMLReporter(config)
reporter.write(result, "styled-report.html")
```

---

## Testing Reporters

Use the SDK testing utilities.

```python
from truthound.reporters.sdk.testing import ReporterTestCase

class TestMyReporter(ReporterTestCase):
    reporter_class = MyCustomReporter

    def test_render_output(self):
        # Create sample report
        report = self.create_sample_report(num_issues=5)

        # Render
        output = self.reporter.render(report)

        # Assertions
        self.assertIn("Issues", output)
        self.assertValidOutput(output)
```

## See Also

- [CLI Output Formats](../cli/common/output-formats.md) - CLI output options
- [HTML Report Guide](../guides/html-reports.md) - Customizing HTML reports
- [CI/CD Integration](../guides/cicd.md) - Using reporters in pipelines
