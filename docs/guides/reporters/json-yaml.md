# JSON & YAML

Use JSON when machines consume the output. Use YAML when humans need a
structured export that is still easy to read and diff.

## JSON Reporter

`json` is part of the built-in reporter factory surface.

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json", indent=2)
json_output = reporter.render(run_result)
```

### Common options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indent` | `int \| None` | `2` | Pretty-print indentation |
| `sort_keys` | `bool` | `False` | Sort keys in output |
| `ensure_ascii` | `bool` | `False` | Escape non-ASCII characters |
| `include_null_values` | `bool` | `True` | Keep null/empty fields |
| `date_format` | `str` | `"iso"` | `iso` or `timestamp` |

### Example workflow

```python
from truthound.reporters import get_reporter

reporter = get_reporter("json", indent=2)
reporter.write(run_result, "artifacts/validation.json")
```

The JSON payload is presentation-oriented and built from the canonical
`ValidationRunResult`.

## YAML Reporter

YAML output is provided by the SDK template reporter:

```python
from truthound.reporters.sdk.templates import YAMLReporter

reporter = YAMLReporter(indent=2, include_passed=False)
yaml_output = reporter.render(run_result)
```

### Common options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_flow_style` | `bool` | `False` | Inline nested structures |
| `indent` | `int` | `2` | YAML indentation |
| `include_passed` | `bool` | `False` | Include passed compatibility rows |
| `sort_keys` | `bool` | `False` | Sort dictionary keys |

### Output shape

The YAML reporter emits a compatibility-oriented document rooted at
`validation_result:`. This is expected behavior for the SDK template and does
not change the canonical 3.0 runtime contract, which remains
`ValidationRunResult`.

```yaml
validation_result:
  run_id: run_20260321_120000_abcd1234
  data_asset: customers.csv
  status: failure
  issues:
    - validator: null
      column: email
      severity: critical
      message: Found null values
```

## NDJSON Reporter

For line-oriented event pipelines, use the SDK template reporter:

```python
from truthound.reporters.sdk.templates import NDJSONReporter

reporter = NDJSONReporter()
ndjson_output = reporter.render(run_result)
```

Good fits:

- log shipping
- streaming ingestion
- Splunk / ELK style pipelines
- append-only audit artifacts

## File Output

All of these reporters can write directly to disk:

```python
from truthound.reporters import get_reporter
from truthound.reporters.sdk.templates import YAMLReporter

get_reporter("json").write(run_result, "validation.json")
YAMLReporter().write(run_result, "validation.yaml")
```

## Dependency Notes

- `JSONReporter` is available with the core reporter surface.
- `YAMLReporter` requires `PyYAML`.
- `NDJSONReporter` ships with the reporter SDK templates and has no extra
  external dependency beyond the SDK surface.

## Choose The Right Format

| Format | Use when |
|--------|----------|
| JSON | another system parses the output |
| YAML | reviewers need a readable structured export |
| NDJSON | events must be streamed or appended line by line |

## See Also

- [Reporters Guide](index.md)
- [HTML & Markdown](html-markdown.md)
- [CI/CD Reporters](ci-reporters.md)
- [Python API: Reporters](../../python-api/reporters.md)
