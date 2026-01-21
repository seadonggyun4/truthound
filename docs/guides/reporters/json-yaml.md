# JSON & YAML Reporters

Reporters that output validation results in JSON and YAML formats.

## JSON Reporter

### Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json")
json_output = reporter.render(validation_result)
```

### Configuration Options

`JSONReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indent` | `int \| None` | `2` | JSON indentation (None = single line) |
| `sort_keys` | `bool` | `False` | Sort keys |
| `ensure_ascii` | `bool` | `False` | Use ASCII characters only |
| `include_null_values` | `bool` | `True` | Include null values |
| `date_format` | `str` | `"iso"` | Date format ("iso" \| "timestamp") |

### Usage Examples

#### Basic JSON Output

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json", indent=2)
output = reporter.render(result)
```

Output example:
```json
{
  "run_id": "abc123-def456",
  "data_asset": "customer_data.csv",
  "status": "failure",
  "run_time": "2024-01-15T10:30:45",
  "success": false,
  "statistics": {
    "total_validators": 10,
    "passed_validators": 8,
    "failed_validators": 2,
    "pass_rate": 0.8,
    "execution_time_ms": 150
  },
  "results": [
    {
      "validator_name": "NullValidator",
      "column": "email",
      "success": false,
      "severity": "critical",
      "issue_type": "null_values",
      "count": 5,
      "message": "Found 5 null values"
    }
  ]
}
```

#### Compact JSON (Single Line)

```python
reporter = get_reporter("json", indent=None)
compact_output = reporter.render(result)

# Or use method
reporter = get_reporter("json")
compact_output = reporter.render_compact(result)
```

#### NDJSON (Newline Delimited JSON)

Useful for integration with log collection systems (ELK, Splunk):

```python
reporter = get_reporter("json")
ndjson_output = reporter.render_lines(result)
```

Output:
```json
{"type":"metadata","run_id":"abc123","data_asset":"data.csv"}
{"type":"result","validator":"NullValidator","column":"email","success":false}
{"type":"result","validator":"RangeValidator","column":"age","success":false}
```

### Date Format

```python
# ISO 8601 format (default)
reporter = get_reporter("json", date_format="iso")
# Output: "2024-01-15T10:30:45"

# Unix timestamp
reporter = get_reporter("json", date_format="timestamp")
# Output: 1705314645
```

---

## YAML Reporter

### Basic Usage

```python
from truthound.reporters import get_reporter

reporter = get_reporter("yaml")
yaml_output = reporter.render(validation_result)
```

### Configuration Options

`YAMLReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_flow_style` | `bool` | `False` | Use flow style |
| `indent` | `int` | `2` | Indentation size |
| `include_passed` | `bool` | `False` | Include passed validators |
| `sort_keys` | `bool` | `False` | Sort keys |

### Usage Examples

```python
from truthound.reporters import get_reporter

reporter = get_reporter("yaml", include_passed=False)
yaml_output = reporter.render(result)
```

Output example:
```yaml
validation_result:
  run_id: abc123-def456
  data_asset: customer_data.csv
  status: failure
  run_time: '2024-01-15T10:30:45'
  summary:
    total_validators: 10
    passed: 8
    failed: 2
    pass_rate: '80.0%'
  severity_counts:
    critical: 1
    high: 1
  issues:
    - validator: NullValidator
      column: email
      severity: critical
      issue_type: null_values
      message: Found 5 null values
      count: 5
    - validator: RangeValidator
      column: age
      severity: high
      issue_type: out_of_range
      message: 3 values out of range
      count: 3
```

### Dependencies

YAML Reporter requires the PyYAML library:

```bash
pip install pyyaml
```

---

## NDJSON Reporter (SDK)

A dedicated NDJSON reporter provided by the SDK.

### Basic Usage

```python
from truthound.reporters.sdk import NDJSONReporter

reporter = NDJSONReporter()
ndjson_output = reporter.render(validation_result)
```

### Configuration Options

`NDJSONReporterConfig` provides the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_passed` | `bool` | `False` | Include passed validators |
| `include_metadata` | `bool` | `True` | Include metadata line |
| `compact` | `bool` | `True` | Use compact JSON |

### Output Format

```json
{"type":"metadata","run_id":"abc123","data_asset":"data.csv","status":"failure","total_validators":10}
{"type":"result","validator":"NullValidator","column":"email","success":false,"severity":"critical","count":5,"message":"Found 5 null values"}
{"type":"result","validator":"RangeValidator","column":"age","success":false,"severity":"high","count":3,"message":"3 values out of range"}
```

---

## File Output

All reporters support file output:

```python
# Method 1: write() method
reporter = get_reporter("json")
path = reporter.write(result, "report.json")

# Method 2: report() method (render + optional file output)
output = reporter.report(result, path="report.json")

# Render as bytes
bytes_output = reporter.render_to_bytes(result)
```

## API Reference

### JSONReporter

```python
class JSONReporter(ValidationReporter[JSONReporterConfig]):
    """JSON format reporter."""

    name = "json"
    file_extension = ".json"
    content_type = "application/json"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as JSON."""
        ...

    def render_compact(self, data: ValidationResult) -> str:
        """Render compact JSON (single line)."""
        ...

    def render_lines(self, data: ValidationResult) -> str:
        """Render in NDJSON format."""
        ...
```

### YAMLReporter

```python
class YAMLReporter(ValidationReporter[YAMLReporterConfig]):
    """YAML format reporter."""

    name = "yaml"
    file_extension = ".yaml"
    content_type = "application/yaml"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as YAML."""
        ...
```

### NDJSONReporter

```python
class NDJSONReporter(ValidationReporter[NDJSONReporterConfig]):
    """NDJSON (Newline Delimited JSON) reporter."""

    name = "ndjson"
    file_extension = ".ndjson"
    content_type = "application/x-ndjson"

    def render(self, data: ValidationResult) -> str:
        """Render validation result as NDJSON."""
        ...
```
