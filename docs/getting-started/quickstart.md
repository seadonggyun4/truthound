# Quick Start

Get started with Truthound in 5 minutes!

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

Output:
```
Schema saved to schema.yaml
  Columns: 5
  Rows: 100
```

### 2. Check Data Quality

```bash
truthound check sample_data.csv
```

Output:
```
Data Quality Report
===================
File: sample_data.csv
Rows: 100
Columns: 5

Issues Found: 4

  MEDIUM  null_check       name: 25 null values (25.0%)
  HIGH    format_check     email: 25 invalid emails
  HIGH    range_check      age: Values outside valid range
  MEDIUM  date_check       created_at: 50 invalid dates
```

### 3. Scan for PII

```bash
truthound scan sample_data.csv
```

Output:
```
PII Scan Report
===============

Potential PII Detected:
  - email: Email addresses (100 rows)
  - name: Person names (75 rows)
```

### 4. Profile Data

```bash
truthound auto-profile sample_data.csv -o profile.json
```

## Python API Quick Start

### Basic Usage

```python
import truthound as th

# Check data quality
report = th.check("sample_data.csv")

if report.has_issues:
    print(f"Found {report.issue_count} issues")
    for issue in report.issues:
        print(f"  {issue.severity}: {issue.message}")
```

### With Schema Validation

```python
# Learn schema from good data
schema = th.learn("reference_data.csv")
schema.save("schema.yaml")

# Validate new data against schema
report = th.check(
    "new_data.csv",
    schema="schema.yaml"
)
```

### Profiling

```python
from truthound import DataProfiler

profiler = DataProfiler()
profile = profiler.profile("sample_data.csv")

print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")

for col in profile.columns:
    print(f"\n{col.name}:")
    print(f"  Type: {col.inferred_type}")
    print(f"  Nulls: {col.null_ratio:.1%}")
    print(f"  Unique: {col.distinct_count}")
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

```yaml
# truthound.yaml
checkpoints:
  - name: daily_validation
    data_source: data/production.csv
    validators: [null, duplicate, range]
    auto_schema: true
    actions:
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        notify_on: failure
```

Run with:
```bash
truthound checkpoint run daily_validation --config truthound.yaml
```

## Next Steps

- [First Validation Tutorial](first-validation.md) - Detailed walkthrough
- [Validators Guide](../user-guide/validators.md) - All available validators
- [CI/CD Integration](../user-guide/ci-cd.md) - Pipeline setup
