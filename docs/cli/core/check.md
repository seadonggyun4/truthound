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

### Core Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--validators` | `-v` | None | Comma-separated list of validators to run (runs all validators when not specified) |
| `--exclude-columns` | `-e` | None | Columns to exclude from all validators (comma-separated) |
| `--validator-config` | `-vc` | None | Validator configuration as JSON string or path to JSON/YAML file |
| `--min-severity` | `-s` | None | Minimum severity level to report (low, medium, high, critical) |
| `--schema` | | None | Schema file for validation |
| `--auto-schema` | | `false` | Auto-learn and cache schema |
| `--format` | `-f` | `console` | Output format (console, json, html) |
| `--output` | `-o` | None | Output file path (required for html format) |
| `--strict` | | `false` | Exit with code 1 if issues found |

### Result Format Options (VE-1)

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--result-format` | `--rf` | `summary` | Detail level: `boolean_only`, `basic`, `summary`, `complete` |
| `--include-unexpected-rows` | | `false` | Include failing rows in output (requires `--rf complete`) |
| `--max-unexpected-rows` | | `1000` | Maximum number of unexpected rows to include |
| `--partial-unexpected-count` | | `20` | Maximum number of unexpected values in partial list (BASIC+) |
| `--include-unexpected-index` | | `false` | Include row index for each unexpected value in results |
| `--return-debug-query` | | `false` | Include Polars debug query expression in results (COMPLETE level) |

### Exception Handling Options (VE-5)

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--catch-exceptions` / `--no-catch-exceptions` | | `true` | Isolate validator exceptions instead of aborting |
| `--max-retries` | | `0` | Number of retries for transient failures |
| `--show-exceptions` | | `false` | Display exception details in output |

### Execution Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--parallel` / `--no-parallel` | | `false` | Enable DAG-based parallel execution with dependency-aware scheduling |
| `--max-workers` | | Auto | Maximum worker threads (only with `--parallel`). Defaults to `min(32, cpu_count + 4)` |
| `--pushdown` / `--no-pushdown` | | Auto | Enable query pushdown for SQL data sources. Auto-detects by default |
| `--use-engine` / `--no-use-engine` | | `false` | Use execution engine for validation (experimental) |

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

### Result Format Control (VE-1)

Control the detail level of validation output:

```bash
# Quick pass/fail check (fastest)
truthound check data.csv --rf boolean_only

# Basic with sample values
truthound check data.csv --rf basic

# Full detail with unexpected rows and debug queries
truthound check data.csv --rf complete --include-unexpected-rows

# Limit unexpected rows
truthound check data.csv --rf complete --include-unexpected-rows --max-unexpected-rows 500
```

### Parallel Execution

Enable DAG-based parallel execution for large validator sets:

```bash
# Enable parallel execution with automatic worker count
truthound check data.csv --parallel

# Control the number of worker threads
truthound check data.csv --parallel --max-workers 8

# Combine with other options
truthound check data.csv --parallel --max-workers 4 --rf summary --strict
```

Validators are organized into dependency levels (Schema → Completeness → Uniqueness → Distribution → Referential) and executed concurrently within each level.

### Advanced Result Format Control

Fine-tune the detail level of validation results:

```bash
# Control partial unexpected list size
truthound check data.csv --rf basic --partial-unexpected-count 50

# Include row indices for unexpected values
truthound check data.csv --rf summary --include-unexpected-index

# Include Polars debug query in results (for troubleshooting)
truthound check data.csv --rf complete --return-debug-query

# All fine-grained options combined
truthound check data.csv --rf complete \
    --include-unexpected-rows \
    --max-unexpected-rows 500 \
    --partial-unexpected-count 100 \
    --include-unexpected-index \
    --return-debug-query
```

### Query Pushdown

For SQL data sources, enable server-side validation:

```bash
# Auto-detect pushdown capability
truthound check data.csv --pushdown

# Explicitly disable pushdown
truthound check data.csv --no-pushdown
```

### Exception Handling (VE-5)

Control error behavior during validation:

```bash
# Retry transient failures up to 3 times
truthound check data.csv --max-retries 3

# Strict mode: abort on first exception
truthound check data.csv --no-catch-exceptions

# Show exception details in output
truthound check data.csv --show-exceptions

# Combined: resilient mode with visibility
truthound check data.csv --catch-exceptions --max-retries 2 --show-exceptions
```

### Column Exclusion

Exclude specific columns from all validators globally:

```bash
# Exclude columns that are expected to have non-unique values
truthound check users.csv --exclude-columns first_name,last_name

# Combine with specific validators
truthound check users.csv -v null,unique,schema -e first_name
```

The `--exclude-columns` option applies to every validator in the run. This is useful when certain columns are known to violate constraints by design (e.g., `first_name` is not expected to be unique).

### Validator Configuration

Pass per-validator configuration via JSON string or file:

```bash
# Inline JSON: exclude first_name from uniqueness checks only
truthound check users.csv --validator-config '{"unique": {"exclude_columns": ["first_name"]}}'

# JSON file
truthound check users.csv --validator-config validator_config.json

# YAML file (requires PyYAML)
truthound check users.csv --validator-config validator_config.yaml
```

The `--validator-config` option accepts a JSON object mapping validator names to their configuration dictionaries. Each validator's configuration is passed as keyword arguments to the validator constructor, enabling fine-grained control over individual validator behavior without affecting others.

**Example `validator_config.json`:**

```json
{
  "unique": {
    "exclude_columns": ["first_name", "last_name"]
  },
  "range": {
    "columns": ["age", "price"]
  }
}
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
