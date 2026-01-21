# Reporter SDK

Truthound provides a comprehensive SDK for custom reporter development.

## Overview

The Reporter SDK provides the following features:

- **Mixins**: Reusable common functionality (formatting, aggregation, filtering, etc.)
- **Builder**: Create reporters with decorators and builder patterns
- **Templates**: Pre-defined reporter templates (CSV, YAML, JUnit, etc.)
- **Schema**: Output format validation
- **Testing**: Testing utilities and mock data generation

---

## Quick Start

### Create Simple Reporter with Decorator

```python
from truthound.reporters.sdk import create_reporter

@create_reporter("my_format", extension=".myf")
def render_my_format(result, config):
    return f"Status: {result.status.value}"

# Usage
from truthound.reporters import get_reporter
reporter = get_reporter("my_format")
output = reporter.render(validation_result)
```

### Full Reporter with Mixins

```python
from truthound.reporters.sdk import (
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
)
from truthound.reporters.base import ValidationReporter, ReporterConfig

class MyReporterConfig(ReporterConfig):
    custom_option: str = "default"

class MyReporter(FormattingMixin, AggregationMixin, FilteringMixin, ValidationReporter[MyReporterConfig]):
    name = "my_format"
    file_extension = ".myf"

    @classmethod
    def _default_config(cls):
        return MyReporterConfig()

    def render(self, data):
        # Use mixin methods
        issues = self.filter_by_severity(data, min_severity="medium")
        grouped = self.group_by_column(issues)
        return self.format_as_table(grouped)
```

---

## Mixins

The SDK provides 6 mixins.

### FormattingMixin

Output formatting utilities:

```python
from truthound.reporters.sdk import FormattingMixin

class MyReporter(FormattingMixin, ValidationReporter):
    def render(self, data):
        # Table formatting (ascii, markdown, grid, simple styles)
        rows = [{"name": r.column, "message": r.message} for r in data.results]
        table = self.format_as_table(rows, style="markdown")

        # Number formatting
        rate = self.format_percentage(data.statistics.pass_rate)

        # Date formatting
        date = self.format_datetime(data.run_time)

        # Byte size formatting
        size = self.format_bytes(1024000)  # "1000.0 KB"

        return f"{table}\nPass Rate: {rate}"
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `format_as_table(rows, columns, style)` | Format data as table (style: ascii/markdown/grid/simple) |
| `format_percentage(value, precision)` | Format percentage (e.g., "85.5%") |
| `format_number(value, precision)` | Format number (thousands separator) |
| `format_datetime(dt, format)` | Format date/time |
| `format_duration(seconds)` | Format execution time (e.g., "2h 30m 15s") |
| `format_bytes(size)` | Format byte size (e.g., "1.5 MB") |
| `format_relative_time(dt)` | Format relative time (e.g., "5 minutes ago") |
| `truncate(text, max_length, suffix)` | Limit text length |
| `indent(text, prefix)` | Indent text |
| `wrap(text, width)` | Wrap text lines |

### AggregationMixin

Data aggregation utilities:

```python
from truthound.reporters.sdk import AggregationMixin

class MyReporter(AggregationMixin, ValidationReporter):
    def render(self, data):
        # Group by column
        by_column = self.group_by_column(data.results)

        # Group by severity
        by_severity = self.group_by_severity(data.results)

        # Group by validator
        by_validator = self.group_by_validator(data.results)

        # Calculate statistics
        stats = self.get_summary_stats(data)

        return self.format_groups(by_severity)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `group_by_column(results)` | Group results by column |
| `group_by_severity(results)` | Group results by severity |
| `group_by_validator(results)` | Group results by validator |
| `group_by(results, key)` | Group by custom key function |
| `get_summary_stats(result)` | Calculate statistics (pass_rate, counts, etc.) |
| `count_by_severity(results)` | Count by severity |
| `count_by_column(results)` | Count by column |

### FilteringMixin

Data filtering utilities:

```python
from truthound.reporters.sdk import FilteringMixin

class MyReporter(FilteringMixin, ValidationReporter):
    def render(self, data):
        # Filter by severity
        critical = self.filter_by_severity(data.results, min_severity="critical")

        # Failed items only
        failed = self.filter_failed(data.results)

        # Specific columns only
        email_issues = self.filter_by_column(data.results, include_columns=["email"])

        # Specific validators only
        null_issues = self.filter_by_validator(data.results, include_validators=["NullValidator"])

        # Sort by severity
        sorted_results = self.sort_by_severity(failed)

        return self.format_issues(sorted_results)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `filter_by_severity(results, min_severity, max_severity)` | Filter by severity range |
| `filter_failed(results)` | Filter failed results only |
| `filter_passed(results)` | Filter passed results only |
| `filter_by_column(results, include_columns, exclude_columns)` | Filter by specific columns |
| `filter_by_validator(results, include_validators, exclude_validators)` | Filter by specific validators |
| `sort_by_severity(results, ascending)` | Sort by severity |
| `sort_by_column(results, ascending)` | Sort by column name |
| `limit(results, count, offset)` | Limit result count |

### SerializationMixin

Serialization utilities:

```python
from truthound.reporters.sdk import SerializationMixin

class MyReporter(SerializationMixin, ValidationReporter):
    def render(self, data):
        # To JSON string
        as_json = self.to_json(data, indent=2)

        # To CSV string
        rows = [{"name": "col1", "count": 10}]
        as_csv = self.to_csv(rows, columns=["name", "count"])

        # Create XML element
        xml_elem = self.to_xml_element(
            "issue",
            value="message",
            attributes={"severity": "high"}
        )

        return as_json
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `to_json(data, indent, sort_keys)` | Serialize to JSON string |
| `to_csv(rows, columns, delimiter)` | Serialize to CSV string |
| `to_xml_element(tag, value, attributes)` | Create XML element string |

### TemplatingMixin

Template rendering utilities:

```python
from truthound.reporters.sdk import TemplatingMixin

class MyReporter(TemplatingMixin, ValidationReporter):
    template_string = """
    Report: {{ data.data_asset }}
    Status: {{ data.status.value }}
    {% for issue in data.issues %}
    - {{ issue.message }}
    {% endfor %}
    """

    def render(self, data):
        return self.render_template(self.template_string, data=data)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `render_template(template, context)` | Render Jinja2 template |
| `render_template_file(path, context)` | Render file-based template |
| `interpolate(template, context)` | Simple string interpolation (no Jinja2 required) |

### StreamingMixin

Streaming output utilities:

```python
from truthound.reporters.sdk import StreamingMixin

class MyReporter(StreamingMixin, ValidationReporter):
    def render(self, data):
        # Generate in chunks
        for chunk in self.stream_results(data.results, chunk_size=100):
            yield self.format_chunk(chunk)

    def render_lines(self, data):
        # Line-by-line streaming
        formatter = lambda r: f"{r.validator_name}: {r.message}"
        return self.render_streaming(data.results, formatter)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `stream_results(results, chunk_size)` | Chunk iterator |
| `stream_lines(results, formatter)` | Line iterator |
| `render_streaming(results, formatter)` | Combine streaming results to string |

---

## Builder

### @create_reporter Decorator

Convert function to reporter:

```python
from truthound.reporters.sdk import create_reporter

@create_reporter(
    name="simple",
    extension=".txt",
    content_type="text/plain"
)
def render_simple(result, config):
    """Simple text reporter."""
    lines = [
        f"Data Asset: {result.data_asset}",
        f"Status: {result.status.value}",
        f"Pass Rate: {result.pass_rate * 100:.1f}%",
    ]
    return "\n".join(lines)

# Automatically registered
from truthound.reporters import get_reporter
reporter = get_reporter("simple")
```

### @create_validation_reporter Decorator

Include full ValidationReporter functionality:

```python
from truthound.reporters.sdk import create_validation_reporter
from truthound.reporters.base import ReporterConfig

class MyConfig(ReporterConfig):
    prefix: str = ">"
    include_timestamp: bool = True

@create_validation_reporter(
    name="prefixed",
    extension=".txt",
    config_class=MyConfig
)
def render_prefixed(result, config):
    lines = []
    if config.include_timestamp:
        lines.append(f"{config.prefix} Time: {result.run_time}")
    lines.append(f"{config.prefix} Status: {result.status.value}")
    return "\n".join(lines)
```

### ReporterBuilder

Fluent builder pattern:

```python
from truthound.reporters.sdk import ReporterBuilder

# ReporterBuilder takes name in constructor
reporter_class = (
    ReporterBuilder("custom")
    .with_extension(".custom")
    .with_content_type("text/plain")
    .with_mixin(FormattingMixin)
    .with_mixin(FilteringMixin)
    .with_renderer(lambda self, data: f"Status: {data.status.value}")
    .build()
)

# Create instance
instance = reporter_class()
output = instance.render(validation_result)
```

**Builder Methods:**

| Method | Description |
|--------|-------------|
| `ReporterBuilder(name)` | Create builder with reporter name |
| `with_extension(ext)` | Set file extension |
| `with_content_type(type)` | Set MIME type |
| `with_mixin(mixin_class)` | Add mixin |
| `with_mixins(*mixins)` | Add multiple mixins |
| `with_config_class(cls)` | Specify config class |
| `with_renderer(func)` | Specify render function (takes self, data as arguments) |
| `with_post_processor(func)` | Add post-processor function |
| `with_attribute(name, value)` | Add class attribute |
| `register_as(name)` | Specify factory registration name |
| `build()` | Create reporter class |

---

## Templates

The SDK provides pre-defined reporter templates.

### CSVReporter

```python
from truthound.reporters.sdk import CSVReporter

reporter = CSVReporter(
    delimiter=",",
    include_header=True,
    include_passed=False,
    quoting="minimal"  # minimal, all, none, nonnumeric
)
csv_output = reporter.render(result)
```

**CSVReporterConfig Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `delimiter` | `str` | `","` | Field delimiter |
| `include_header` | `bool` | `True` | Include header row |
| `include_passed` | `bool` | `False` | Include passed items |
| `quoting` | `str` | `"minimal"` | Quoting style |
| `columns` | `list[str]` | `None` | Columns to include (None=all) |

### YAMLReporter

```python
from truthound.reporters.sdk import YAMLReporter

reporter = YAMLReporter(
    default_flow_style=False,
    indent=2,
    include_passed=False,
    sort_keys=False
)
yaml_output = reporter.render(result)
```

**YAMLReporterConfig Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_flow_style` | `bool` | `False` | Use flow style |
| `indent` | `int` | `2` | Indentation size |
| `include_passed` | `bool` | `False` | Include passed items |
| `sort_keys` | `bool` | `False` | Sort keys |

### JUnitXMLReporter

JUnit XML format for CI/CD integration:

```python
from truthound.reporters.sdk import JUnitXMLReporter

reporter = JUnitXMLReporter(
    testsuite_name="Truthound Validation",
    include_stdout=True,
    include_properties=True
)
xml_output = reporter.render(result)
```

**JUnitXMLReporterConfig Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `testsuite_name` | `str` | `"Truthound Validation"` | Test suite name |
| `include_stdout` | `bool` | `True` | Include system-out |
| `include_properties` | `bool` | `True` | Include properties |
| `include_passed` | `bool` | `False` | Include passed tests |

### NDJSONReporter

Newline Delimited JSON (for log collection system integration):

```python
from truthound.reporters.sdk import NDJSONReporter

reporter = NDJSONReporter(
    include_metadata=True,
    include_passed=False,
    compact=True
)
ndjson_output = reporter.render(result)
```

**NDJSONReporterConfig Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_metadata` | `bool` | `True` | Include metadata line |
| `include_passed` | `bool` | `False` | Include passed items |
| `compact` | `bool` | `True` | Compact JSON |

### TableReporter

Text table output:

```python
from truthound.reporters.sdk import TableReporter

reporter = TableReporter(
    style="grid",  # ascii, markdown, grid, simple
    max_width=120,
    include_passed=False
)
table_output = reporter.render(result)
```

**TableReporterConfig Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `style` | `str` | `"ascii"` | Table style |
| `max_width` | `int` | `120` | Maximum width |
| `include_passed` | `bool` | `False` | Include passed items |
| `show_index` | `bool` | `False` | Show index |

---

## Schema Validation

A schema system for output format validation.

### Basic Usage

```python
from truthound.reporters.sdk import validate_output, JSONSchema

# Define schema
schema = JSONSchema(
    required_fields=["status", "data_asset", "issues"],
    field_types={
        "status": str,
        "data_asset": str,
        "issues": list,
        "pass_rate": float,
    }
)

# Validate output
result = validate_output(json_output, schema)
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.message}")
```

### Schema Types

#### JSONSchema

```python
from truthound.reporters.sdk import JSONSchema

schema = JSONSchema(
    required_fields=["status", "data_asset"],
    field_types={"status": str, "issues": list},
    allow_extra_fields=True,
    max_depth=10
)
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `required_fields` | `list[str]` | Required field list |
| `field_types` | `dict[str, type]` | Field types |
| `allow_extra_fields` | `bool` | Allow extra fields |
| `max_depth` | `int` | Maximum nesting depth |

#### XMLSchema

```python
from truthound.reporters.sdk import XMLSchema

schema = XMLSchema(
    root_element="testsuites",
    required_elements=["testsuite", "testcase"],
    required_attributes={"testsuite": ["name", "tests"]},
    validate_dtd=False
)
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `root_element` | `str` | Root element name |
| `required_elements` | `list[str]` | Required element list |
| `required_attributes` | `dict[str, list[str]]` | Required attributes per element |
| `validate_dtd` | `bool` | Validate DTD |

#### CSVSchema

```python
from truthound.reporters.sdk import CSVSchema

schema = CSVSchema(
    required_columns=["validator", "column", "severity", "message"],
    column_types={"severity": str, "count": int},
    allow_extra_columns=True,
    min_rows=0
)
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `required_columns` | `list[str]` | Required column list |
| `column_types` | `dict[str, type]` | Column types |
| `allow_extra_columns` | `bool` | Allow extra columns |
| `min_rows` | `int` | Minimum row count |

#### TextSchema

```python
from truthound.reporters.sdk import TextSchema

schema = TextSchema(
    required_patterns=[r"Status:", r"Pass Rate:"],
    forbidden_patterns=[r"ERROR", r"EXCEPTION"],
    max_length=100000,
    encoding="utf-8"
)
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `required_patterns` | `list[str]` | Required patterns (regex) |
| `forbidden_patterns` | `list[str]` | Forbidden patterns (regex) |
| `max_length` | `int` | Maximum character count |
| `encoding` | `str` | Encoding |

### Schema Registration and Management

```python
from truthound.reporters.sdk import (
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
)

# Register schema
register_schema("my_format", schema)

# Get schema
my_schema = get_schema("my_format")

# Remove schema
unregister_schema("my_format")

# Auto-validate reporter output
is_valid = validate_reporter_output("json", json_output)
```

### Schema Inference and Merging

```python
from truthound.reporters.sdk import infer_schema, merge_schemas

# Infer schema from sample data
inferred = infer_schema(sample_output, format="json")

# Merge multiple schemas
merged = merge_schemas([schema1, schema2], strategy="union")
```

---

## Testing Utilities

Utilities for reporter testing.

### Mock Data Generation

#### create_mock_result

```python
from truthound.reporters.sdk import create_mock_result

# Default mock result
result = create_mock_result()

# Custom settings
result = create_mock_result(
    data_asset="test_data.csv",
    status="failure",
    pass_rate=0.75,
    issue_count=5,
    severity_distribution={"critical": 2, "high": 2, "medium": 1}
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_asset` | `str` | `"test_data.csv"` | Data asset name |
| `status` | `str` | `"failure"` | Validation status |
| `pass_rate` | `float` | `0.8` | Pass rate |
| `issue_count` | `int` | `3` | Issue count |
| `severity_distribution` | `dict` | `None` | Issues per severity |

#### MockResultBuilder

Fluent builder pattern:

```python
from truthound.reporters.sdk import MockResultBuilder

result = (
    MockResultBuilder()
    .with_data_asset("orders.parquet")
    .with_status("failure")
    .with_pass_rate(0.65)
    .add_issue(
        validator="NullValidator",
        column="email",
        severity="critical",
        message="Found 10 null values"
    )
    .add_issue(
        validator="RangeValidator",
        column="age",
        severity="high",
        message="5 values out of range"
    )
    .with_run_time("2024-01-15T10:30:45")
    .build()
)
```

**Builder Methods:**

| Method | Description |
|--------|-------------|
| `with_data_asset(name)` | Set data asset name |
| `with_status(status)` | Set validation status |
| `with_pass_rate(rate)` | Set pass rate |
| `with_run_time(time)` | Set run time |
| `add_issue(...)` | Add issue |
| `add_issues(issues)` | Add multiple issues |
| `build()` | Create MockValidationResult |

#### create_mock_results

Generate multiple results:

```python
from truthound.reporters.sdk import create_mock_results

# Generate 5 random results
results = create_mock_results(count=5)

# Various status distributions
results = create_mock_results(
    count=10,
    status_distribution={"success": 0.7, "failure": 0.3}
)
```

### Assertion Functions

#### General Validation

```python
from truthound.reporters.sdk import assert_valid_output

# Auto-detect and validate output format
assert_valid_output(output, format="json")
assert_valid_output(output, format="xml")
assert_valid_output(output, format="csv")
```

#### JSON Validation

```python
from truthound.reporters.sdk import assert_json_valid

# Basic JSON validation
assert_json_valid(json_output)

# Validate with schema
assert_json_valid(json_output, schema=my_schema)

# Validate required fields
assert_json_valid(json_output, required_fields=["status", "issues"])
```

#### XML Validation

```python
from truthound.reporters.sdk import assert_xml_valid

# Basic XML validation
assert_xml_valid(xml_output)

# Validate root element
assert_xml_valid(xml_output, root_element="testsuites")

# Validate with XSD schema
assert_xml_valid(xml_output, xsd_path="schema.xsd")
```

#### CSV Validation

```python
from truthound.reporters.sdk import assert_csv_valid

# Basic CSV validation
assert_csv_valid(csv_output)

# Validate columns
assert_csv_valid(
    csv_output,
    required_columns=["validator", "column", "severity"],
    min_rows=1
)
```

#### Pattern Matching

```python
from truthound.reporters.sdk import assert_contains_patterns

assert_contains_patterns(
    output,
    patterns=[
        r"Status: (success|failure)",
        r"Pass Rate: \d+\.\d+%",
        r"Total Issues: \d+"
    ]
)
```

### ReporterTestCase

Base class for test cases:

```python
from truthound.reporters.sdk import ReporterTestCase
from truthound.reporters import get_reporter

class TestMyReporter(ReporterTestCase):
    reporter_name = "my_format"

    def test_basic_render(self):
        """Basic rendering test."""
        reporter = get_reporter(self.reporter_name)
        result = self.create_sample_result()

        output = reporter.render(result)

        self.assert_output_valid(output)
        self.assertIn("Status:", output)

    def test_empty_issues(self):
        """Test with no issues."""
        result = self.create_result_with_no_issues()
        output = self.render(result)

        self.assert_output_valid(output)

    def test_edge_cases(self):
        """Edge case tests."""
        for edge_case in self.get_edge_cases():
            with self.subTest(edge_case=edge_case.name):
                output = self.render(edge_case.data)
                self.assert_output_valid(output)
```

**Provided Methods:**

| Method | Description |
|--------|-------------|
| `create_sample_result()` | Create standard sample result |
| `create_result_with_no_issues()` | Create result with no issues |
| `create_result_with_many_issues(n)` | Create result with n issues |
| `get_edge_cases()` | Return edge case list |
| `render(result)` | Render with reporter |
| `assert_output_valid(output)` | Validate output |

### Test Data Generation

```python
from truthound.reporters.sdk import (
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
)

# Standard sample data
sample = create_sample_data()

# Edge case data
edge_cases = create_edge_case_data()
# Returns: empty_result, single_issue, max_severity, unicode_content, ...

# Stress test data
stress = create_stress_test_data(
    issue_count=10000,
    validator_count=100
)
```

### Output Capture and Benchmarking

#### capture_output

```python
from truthound.reporters.sdk import capture_output

# Capture stdout/stderr
with capture_output() as captured:
    reporter.print(result)

print(f"Stdout: {captured.stdout}")
print(f"Stderr: {captured.stderr}")
```

#### benchmark_reporter

```python
from truthound.reporters.sdk import benchmark_reporter, BenchmarkResult

# Benchmark reporter performance
result: BenchmarkResult = benchmark_reporter(
    reporter=get_reporter("json"),
    data=create_stress_test_data(issue_count=1000),
    iterations=100
)

print(f"Mean time: {result.mean_time:.4f}s")
print(f"Std dev: {result.std_dev:.4f}s")
print(f"Min time: {result.min_time:.4f}s")
print(f"Max time: {result.max_time:.4f}s")
print(f"Throughput: {result.throughput:.2f} ops/sec")
```

**BenchmarkResult Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mean_time` | `float` | Mean execution time (seconds) |
| `std_dev` | `float` | Standard deviation |
| `min_time` | `float` | Minimum execution time |
| `max_time` | `float` | Maximum execution time |
| `throughput` | `float` | Throughput per second |
| `iterations` | `int` | Iteration count |

---

## Custom Reporter Registration

### Register with Decorator

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

### Manual Registration

```python
from truthound.reporters.factory import register_reporter

register_reporter("my_custom", MyCustomReporter)

# Usage
reporter = get_reporter("my_custom")
```

---

## API Reference

### SDK Exports

```python
from truthound.reporters.sdk import (
    # Mixins
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
    SerializationMixin,
    TemplatingMixin,
    StreamingMixin,

    # Builder
    ReporterBuilder,
    create_reporter,
    create_validation_reporter,

    # Templates
    CSVReporter,
    YAMLReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,

    # Schema
    ReportSchema,
    JSONSchema,
    XMLSchema,
    CSVSchema,
    TextSchema,
    ValidationResult,
    ValidationError,
    SchemaError,
    validate_output,
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
    infer_schema,
    merge_schemas,

    # Testing
    ReporterTestCase,
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    MockResultBuilder,
    MockValidationResult,
    MockValidatorResult,
    assert_valid_output,
    assert_json_valid,
    assert_xml_valid,
    assert_csv_valid,
    assert_contains_patterns,
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
    capture_output,
    benchmark_reporter,
    BenchmarkResult,
)
```
