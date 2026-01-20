# Quick Start

Get started with Truthound in 5 minutes!

## Supported File Formats

Truthound CLI supports the following file formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | `.csv` | Comma-separated values |
| JSON | `.json` | Newline-delimited JSON (via scan_ndjson) |
| Parquet | `.parquet` | Columnar storage format |
| NDJSON | `.ndjson` | Newline-delimited JSON |
| JSONL | `.jsonl` | JSON Lines (same as NDJSON) |

!!! note "Database and Cloud Data Sources"
    For SQL databases, Spark, or Cloud Data Warehouses (BigQuery, Snowflake, Redshift, Databricks), use the Python API with the `source=` parameter. See [Data Sources](../guides/datasources.md) for details.

## Create Sample Data

First, let's create a sample CSV file:

```python
import polars as pl

df = pl.DataFrame({
    "id": range(1, 101),
    "name": ["Alice", "Bob", None, "Charlie"] * 25,
    "email": ["alice@example.com", "bob@test.com", "invalid-email", "charlie@example.org"] * 25,
    "age": [25, 30, -5, 150] * 25,  # Contains invalid values
    "created_at": ["2024-01-01", "2024-13-45", "2024-02-28", "not-a-date"] * 25,
})

df.write_csv("sample_data.csv")
```

## CLI Quick Start

### 1. Learn Schema

```bash
truthound learn sample_data.csv
```

This command analyzes the data and saves a schema file:

```
Schema saved to schema.yaml
  Columns: 5
  Rows: 100
```

Options:

| Option | Description |
|--------|-------------|
| `-o`, `--output` | Output schema file path (default: `schema.yaml`) |
| `--no-constraints` | Don't infer constraints from data |

### 2. Check Data Quality

```bash
truthound check sample_data.csv
```

Validates data and displays issues found:

```
Truthound Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Column     ┃ Issue              ┃ Count ┃ Severity ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ name       │ null               │    25 │   high   │
│ email      │ invalid_format     │    25 │   high   │
│ age        │ out_of_range       │    50 │  medium  │
└────────────┴────────────────────┴───────┴──────────┘

Summary: 3 issues found
```

Options:

| Option | Description |
|--------|-------------|
| `-v`, `--validators` | Comma-separated list of validators to run |
| `-s`, `--min-severity` | Minimum severity level (`low`, `medium`, `high`, `critical`) |
| `--schema` | Schema file for validation |
| `--auto-schema` | Auto-learn and cache schema (zero-config mode) |
| `-f`, `--format` | Output format (`console`, `json`, `html`) |
| `-o`, `--output` | Output file path |
| `--strict` | Exit with code 1 if issues are found |

!!! note "HTML format requires jinja2"
    Install with: `pip install truthound[reports]` or `pip install jinja2`

### 3. Scan for PII

```bash
truthound scan sample_data.csv
```

Detects personally identifiable information:

```
Truthound PII Scan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Column     ┃ PII Type      ┃ Count ┃ Confidence ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ email      │ Email Address │    75 │    99%     │
│ name       │ Person Name   │    75 │    85%     │
└────────────┴───────────────┴───────┴────────────┘

Warning: Found 2 columns with potential PII
```

Options:

| Option | Description |
|--------|-------------|
| `-f`, `--format` | Output format (`console`, `json`, `html`) |
| `-o`, `--output` | Output file path |

!!! note "HTML format requires jinja2"
    Install with: `pip install truthound[reports]` or `pip install jinja2`

!!! tip "Console output is the default"
    If no format is specified, `console` format is used with Rich formatting for better readability.

### 4. Profile Data

```bash
truthound profile sample_data.csv
```

Generates a statistical profile:

```
Truthound Profile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: sample_data.csv
Rows: 100 | Columns: 5 | Size: 4.4 KB

┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━┓
┃ Column     ┃ Type   ┃ Nulls ┃ Unique ┃ Min ┃ Max ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━┩
│ id         │ Int64  │  0.0% │   100% │   1 │ 100 │
│ name       │ String │ 25.0% │     4% │   - │   - │
│ email      │ String │  0.0% │     4% │   - │   - │
│ age        │ Int64  │  0.0% │     4% │  -5 │ 150 │
│ created_at │ String │  0.0% │     4% │   - │   - │
└────────────┴────────┴───────┴────────┴─────┴─────┘
```

Options:

| Option | Description |
|--------|-------------|
| `-f`, `--format` | Output format (`console`, `json`) |
| `-o`, `--output` | Output file path |

!!! note "Basic vs Advanced Profiling"
    The `profile` command provides basic statistics. For advanced profiling with pattern detection and type inference, use `auto-profile` instead.

### 5. Advanced Profiling (auto-profile)

For comprehensive profiling with pattern detection and type inference:

```bash
truthound auto-profile sample_data.csv -f json -o profile.json
```

Options:

| Option | Description |
|--------|-------------|
| `-f`, `--format` | Output format (`console`, `json`, `yaml`) |
| `-o`, `--output` | Output file path |
| `--patterns/--no-patterns` | Include pattern detection (default: enabled) |
| `--correlations/--no-correlations` | Include correlation analysis (default: disabled) |
| `-s`, `--sample` | Sample size for profiling (default: all rows) |
| `--top-n` | Number of top/bottom values to include (default: 10) |

## Python API Quick Start

### Basic Usage

```python
import truthound as th

# Check data quality
report = th.check("sample_data.csv")

if report.has_issues:
    print(f"Found {len(report.issues)} issues")
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.column}: {issue.issue_type}")
else:
    print("No issues found!")

# Print formatted report (uses Rich for pretty output)
report.print()

# Or get string representation
print(report)
```

### With Schema Validation

```python
import truthound as th
from truthound.schema import learn

# Learn schema from good data
schema = learn("reference_data.csv")
schema.save("schema.yaml")

# Validate new data against schema
report = th.check(
    "new_data.csv",
    schema="schema.yaml"
)
```

### Profiling

```python
import truthound as th

# Basic profiling - returns ProfileReport
profile = th.profile("sample_data.csv")

print(f"Source: {profile.source}")
print(f"Rows: {profile.row_count:,}")
print(f"Columns: {profile.column_count}")
print(f"Size: {profile.size_bytes:,} bytes")

# Column details
for col in profile.columns:
    print(f"\n{col['name']} ({col['dtype']}):")
    print(f"  Nulls: {col['null_pct']}")
    print(f"  Unique: {col['unique_pct']}")
    if col.get('min') and col.get('min') != "-":
        print(f"  Range: [{col['min']}, {col['max']}]")

# Print formatted report (uses Rich for pretty output)
profile.print()
```

### Advanced Profiling with Pattern Detection

For comprehensive profiling including pattern detection and type inference:

```python
from truthound.profiler import DataProfiler, ProfilerConfig

# Configure profiler
config = ProfilerConfig(
    include_patterns=True,
    include_correlations=False,
    sample_size=None,  # Use all rows
    top_n_values=10,
)

profiler = DataProfiler(config=config)

# Profile data
import polars as pl
lf = pl.scan_csv("sample_data.csv")
profile_result = profiler.profile(lf, name="sample", source="sample_data.csv")

# Access detailed profile
print(f"Columns: {profile_result.column_count}")
for col in profile_result.columns:
    print(f"{col.name}: {col.inferred_type.value}")
    if col.detected_patterns:
        print(f"  Patterns: {[p.pattern for p in col.detected_patterns[:3]]}")
```

Or use the CLI for advanced profiling:

```bash
truthound auto-profile sample_data.csv -f json -o profile.json
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on: [push, pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Truthound
        run: pip install truthound

      - name: Check Data Quality
        run: truthound check data/*.csv --strict
```

### Checkpoint Configuration

Create a checkpoint configuration file to define validation pipelines:

```yaml
# truthound.yaml
checkpoints:
- name: daily_data_validation
  data_source: data/production.csv
  validators:
  - 'null'
  - duplicate
  - range
  - regex
  validator_config:
    regex:
      patterns:
        email: ^[\w.+-]+@[\w-]+\.[\w.-]+$
        product_code: ^[A-Z]{2,4}[-_][0-9]{3,6}$
        phone: ^(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$
    range:
      columns:
        age:
          min_value: 0
          max_value: 150
        price:
          min_value: 0
  min_severity: medium
  auto_schema: true
  tags:
    environment: production
    team: data-platform
  actions:
  - type: store_result
    store_path: ./truthound_results
    partition_by: date
  - type: update_docs
    site_path: ./truthound_docs
    include_history: true
  - type: slack
    webhook_url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
    notify_on: failure
    channel: '#data-quality'
  triggers:
  - type: schedule
    interval_hours: 24
    run_on_weekdays: [0, 1, 2, 3, 4]
```

Run with:

```bash
truthound checkpoint run daily_data_validation --config truthound.yaml
```

Or run ad-hoc without a config file:

```bash
truthound checkpoint run quick_check \
    --data data/production.csv \
    --validators null,range \
    --strict \
    --slack https://hooks.slack.com/services/...
```

### Checkpoint CLI Options

| Option | Description |
|--------|-------------|
| `-c`, `--config` | Checkpoint configuration file (YAML/JSON) |
| `-d`, `--data` | Override data source path |
| `-v`, `--validators` | Override validators (comma-separated) |
| `-o`, `--output` | Output file for results (JSON) |
| `-f`, `--format` | Output format (`console`, `json`) |
| `--strict` | Exit with code 1 if issues are found |
| `--store` | Store results to directory |
| `--slack` | Slack webhook URL for notifications |
| `--webhook` | Webhook URL for notifications |
| `--github-summary` | Write GitHub Actions job summary |

## Next Steps

- [First Validation Tutorial](first-validation.md) - Detailed walkthrough
- [Validators Guide](../guides/validators.md) - All 289 built-in validators
- [CI/CD Integration](../guides/ci-cd.md) - Advanced pipeline setup
