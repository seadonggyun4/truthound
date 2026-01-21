# truthound check

Validate data quality in a file. This command runs validators against your data and reports any issues found.

## Synopsis

```bash
truthound check <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--validators` | `-v` | None | Comma-separated list of validators to run (runs all validators when not specified) |
| `--min-severity` | `-s` | None | Minimum severity level to report (low, medium, high, critical) |
| `--schema` | | None | Schema file for validation |
| `--auto-schema` | | `false` | Auto-learn and cache schema |
| `--format` | `-f` | `console` | Output format (console, json, html) |
| `--output` | `-o` | None | Output file path (required for html format) |
| `--strict` | | `false` | Exit with code 1 if issues found |

## Description

The `check` command validates data quality by running a suite of validators:

- **Completeness**: Null values, missing data
- **Uniqueness**: Duplicates, primary key violations
- **Consistency**: Type mismatches, format violations
- **Validity**: Range checks, pattern matching
- **Schema**: Column presence, data type compliance

## Examples

### Basic Validation

Run all validators with default settings:

```bash
truthound check data.csv
```

Output:
```
Data Quality Report
===================
File: data.csv
Rows: 1000
Columns: 5

Issues Found: 3

  HIGH    null_check       email: 50 null values (5.0%)
  MEDIUM  range_check      age: 10 values outside range [0, 120]
  LOW     duplicate_check  id: 2 duplicate values
```

### Specific Validators

Run only selected validators:

```bash
truthound check data.csv -v null,duplicate,range
```

### Severity Filter

Report only high and critical issues:

```bash
truthound check data.csv --min-severity high
```

### Schema Validation

Validate against a predefined schema:

```bash
truthound check data.csv --schema schema.yaml
```

### Auto-Schema Mode

Automatically learn and cache schema on first run:

```bash
truthound check data.csv --auto-schema
```

- First run: Learns schema and caches to `.truthound/schema_cache/`
- Subsequent runs: Validates against cached schema

### Strict Mode (CI/CD)

Exit with code 1 if any issues are found:

```bash
truthound check data.csv --strict
```

### Output Formats

```bash
# Console (default)
truthound check data.csv

# JSON output
truthound check data.csv --format json -o report.json

# HTML report (requires pip install truthound[reports])
truthound check data.csv --format html -o report.html
```

!!! warning "HTML Report Dependency"
    HTML reports require Jinja2. Install with:
    ```bash
    pip install truthound[reports]
    ```

## Available Validators

### Completeness

| Validator | Description |
|-----------|-------------|
| `null` | Check for null values |
| `completeness` | Check completeness ratio |

### Uniqueness

| Validator | Description |
|-----------|-------------|
| `duplicate` | Check for duplicate rows |
| `unique` | Check column uniqueness |

### Validity

| Validator | Description |
|-----------|-------------|
| `range` | Check numeric ranges |
| `pattern` | Check string patterns |
| `email` | Validate email format |
| `phone` | Validate phone format |
| `url` | Validate URL format |
| `date` | Validate date format |

### Consistency

| Validator | Description |
|-----------|-------------|
| `type` | Check data type consistency |
| `format` | Check format consistency |

### Schema

| Validator | Description |
|-----------|-------------|
| `schema` | Validate against schema definition |

## Output Formats

### Console Output

```
Data Quality Report
===================
File: data.csv
Rows: 1000
Columns: 5

Issues Found: 3

  HIGH    null_check       email: 50 null values (5.0%)
  MEDIUM  range_check      age: 10 values outside range [0, 120]
  LOW     duplicate_check  id: 2 duplicate values

Summary:
  Total Issues: 3
  Critical: 0
  High: 1
  Medium: 1
  Low: 1
```

### JSON Output

```json
{
  "file": "data.csv",
  "rows": 1000,
  "columns": 5,
  "passed": false,
  "issues": [
    {
      "validator": "null_check",
      "column": "email",
      "severity": "high",
      "message": "50 null values (5.0%)",
      "details": {
        "null_count": 50,
        "null_ratio": 0.05
      }
    }
  ],
  "summary": {
    "total": 3,
    "critical": 0,
    "high": 1,
    "medium": 1,
    "low": 1
  }
}
```

### HTML Output

Generates an interactive HTML report with:

- Summary dashboard
- Issue breakdown by severity
- Column-level statistics
- Visualizations

## CI/CD Integration

### GitHub Actions

```yaml
- name: Validate Data Quality
  run: truthound check data/*.csv --strict

- name: Generate Report
  if: failure()
  run: truthound check data/*.csv --format html -o report.html

- name: Upload Report
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: data-quality-report
    path: report.html
```

### GitLab CI

```yaml
validate-data:
  script:
    - truthound check data/*.csv --strict --format json -o report.json
  artifacts:
    when: on_failure
    paths:
      - report.json
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (no issues, or issues found without `--strict`) |
| 1 | Issues found with `--strict` flag |
| 2 | Usage error (invalid arguments) |

## Related Commands

- [`learn`](learn.md) - Learn schema from data
- [`scan`](scan.md) - Scan for PII
- [`profile`](profile.md) - Generate data profile

## See Also

- [Python API: th.check()](../../python-api/core-functions.md#thcheck)
- [All Validators Reference](../../guides/validators.md)
- [CI/CD Integration Guide](../../guides/ci-cd.md)
