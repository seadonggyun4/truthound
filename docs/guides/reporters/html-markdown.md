# HTML & Markdown Reporters

Reporters that generate validation reports in HTML and Markdown formats.

## HTML Reporter

### Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("html")
html_output = reporter.render(validation_result)
```

### Configuration Options

`HTMLReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `template` | `str \| None` | `None` | Custom Jinja2 template |
| `template_path` | `str \| None` | `None` | Template file path |
| `inline_css` | `bool` | `True` | Inline CSS in HTML |
| `include_js` | `bool` | `True` | Include JavaScript |
| `theme` | `str` | `"light"` | Theme ("light", "dark") |
| `include_charts` | `bool` | `True` | Include charts |
| `responsive` | `bool` | `True` | Responsive design |

### Usage Examples

#### Basic HTML Report

```python
from truthound.reporters import get_reporter

reporter = get_reporter("html")
html_output = reporter.render(result)

# Save to file
reporter.write(result, "validation_report.html")
```

#### Dark Theme

```python
reporter = get_reporter("html", theme="dark")
html_output = reporter.render(result)
```

#### Custom Template

```python
custom_template = """
<!DOCTYPE html>
<html>
<head><title>{{ title }}</title></head>
<body>
  <h1>{{ result.data_asset }} Validation</h1>
  <p>Status: {{ result.status.value }}</p>
  <ul>
  {% for r in result.results if not r.success %}
    <li>{{ r.validator_name }}: {{ r.message }}</li>
  {% endfor %}
  </ul>
</body>
</html>
"""

reporter = get_reporter("html", template=custom_template)
html_output = reporter.render(result)
```

#### Using Template File

```python
reporter = get_reporter("html", template_path="templates/my_report.html")
html_output = reporter.render(result)
```

### Default Template Features

The default HTML template includes:

1. **Summary Section**: Overall statistics and status
2. **Severity Analysis**: Donut chart (ApexCharts)
3. **Issues Table**: Sortable/filterable table
4. **Detail Information**: Detailed info for each validator result
5. **Responsive Design**: Mobile optimized

### Template Context Variables

Variables available in custom templates:

| Variable | Type | Description |
|----------|------|-------------|
| `result` | `ValidationResult` | Validation result object |
| `title` | `str` | Report title |
| `timestamp` | `str` | Generation time |
| `config` | `HTMLReporterConfig` | Reporter configuration |
| `statistics` | `dict` | Statistics information |

### Dependencies

HTML Reporter uses the Jinja2 template engine:

```bash
pip install jinja2
```

---

## Markdown Reporter

### Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("markdown")
md_output = reporter.render(validation_result)
```

### Configuration Options

`MarkdownReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_toc` | `bool` | `True` | Include table of contents |
| `heading_level` | `int` | `1` | Starting heading level (1-6) |
| `include_badges` | `bool` | `True` | Include shields.io badges |
| `table_style` | `str` | `"github"` | Table style |
| `include_details` | `bool` | `True` | Include details section |
| `include_statistics` | `bool` | `True` | Include statistics section |

### Usage Examples

#### Basic Markdown

```python
from truthound.reporters import get_reporter

reporter = get_reporter("markdown")
md_output = reporter.render(result)
```

Output example:
```markdown
# Validation Report

![Status](https://img.shields.io/badge/Status-FAILED-red)
![Pass Rate](https://img.shields.io/badge/Pass%20Rate-80%25-yellow)

## Summary

| Metric | Value |
|--------|-------|
| Data Asset | customer_data.csv |
| Run ID | abc123-def456 |
| Status | FAILED |
| Total Validators | 10 |
| Passed | 8 |
| Failed | 2 |
| Pass Rate | 80.0% |

## Issues by Severity

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 1 |
| Medium | 0 |
| Low | 0 |

## Failed Validations

### NullValidator - email

- **Severity**: critical
- **Count**: 5
- **Message**: Found 5 null values

### RangeValidator - age

- **Severity**: high
- **Count**: 3
- **Message**: 3 values out of range
```

#### Disable Badges

```python
reporter = get_reporter("markdown", include_badges=False)
```

#### Disable Table of Contents

```python
reporter = get_reporter("markdown", include_toc=False)
```

#### Adjust Heading Level

Heading level can be adjusted when embedding in documents:

```python
# Start from h2
reporter = get_reporter("markdown", heading_level=2)
```

### shields.io Badges

When `include_badges=True`, the following badges are generated:

- **Status Badge**: Pass (green) / Fail (red)
- **Pass Rate Badge**: Color based on rate
- **Issues Badge**: Shows issue count

```markdown
![Status](https://img.shields.io/badge/Status-PASSED-green)
![Pass Rate](https://img.shields.io/badge/Pass%20Rate-100%25-brightgreen)
![Issues](https://img.shields.io/badge/Issues-0-brightgreen)
```

---

## Table Reporter (SDK)

An ASCII/Unicode table reporter provided by the SDK.

### Basic Usage

```python
from truthound.reporters.sdk import TableReporter

reporter = TableReporter()
table_output = reporter.render(validation_result)
```

### Configuration Options

`TableReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `style` | `str` | `"ascii"` | Table style |
| `include_passed` | `bool` | `False` | Include passed validators |
| `max_column_width` | `int` | `50` | Maximum column width |
| `columns` | `list[str]` | `["validator", "column", "severity", "message"]` | Columns to display |
| `sort_by` | `str \| None` | `"severity"` | Sort by |
| `sort_ascending` | `bool` | `False` | Ascending sort |

### Table Styles

#### ASCII (Default)
```
+------------+--------+----------+---------------------------+
| validator  | column | severity | message                   |
+------------+--------+----------+---------------------------+
| NullValidator | email | critical | Found 5 null values    |
| RangeValidator | age | high     | 3 values out of range  |
+------------+--------+----------+---------------------------+
```

#### Markdown
```
| validator  | column | severity | message                   |
|------------|--------|----------|---------------------------|
| NullValidator | email | critical | Found 5 null values    |
| RangeValidator | age | high     | 3 values out of range  |
```

#### Grid (Unicode)
```
╔════════════╤════════╤══════════╤═══════════════════════════╗
║ validator  │ column │ severity │ message                   ║
╠════════════╪════════╪══════════╪═══════════════════════════╣
║ NullValidator │ email │ critical │ Found 5 null values   ║
║ RangeValidator │ age │ high     │ 3 values out of range ║
╚════════════╧════════╧══════════╧═══════════════════════════╝
```

#### Simple
```
validator       column   severity   message
-----------     ------   --------   ---------------------------
NullValidator   email    critical   Found 5 null values
RangeValidator  age      high       3 values out of range
```

---

## File Output

```python
# Save HTML file
html_reporter = get_reporter("html")
html_reporter.write(result, "report.html")

# Save Markdown file
md_reporter = get_reporter("markdown")
md_reporter.write(result, "VALIDATION_REPORT.md")
```

## API Reference

### HTMLReporter

```python
class HTMLReporter(ValidationReporter[HTMLReporterConfig]):
    """HTML format reporter (Jinja2-based)."""

    name = "html"
    file_extension = ".html"
    content_type = "text/html"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as HTML."""
        ...
```

### MarkdownReporter

```python
class MarkdownReporter(ValidationReporter[MarkdownReporterConfig]):
    """Markdown format reporter."""

    name = "markdown"
    file_extension = ".md"
    content_type = "text/markdown"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as Markdown."""
        ...
```

### TableReporter

```python
class TableReporter(ValidationReporter[TableReporterConfig]):
    """ASCII/Unicode table reporter."""

    name = "table"
    file_extension = ".txt"
    content_type = "text/plain"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as table."""
        ...
```
