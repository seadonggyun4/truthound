# First Validation Tutorial

This tutorial walks you through your first data validation with Truthound.

## Objectives

By the end of this tutorial, you will:

1. Understand Truthound's validation workflow
2. Create and validate a dataset
3. Interpret validation results
4. Fix data quality issues
5. Set up continuous validation

## Step 1: Prepare Your Data

Let's create a realistic dataset with common data quality issues:

```python
import polars as pl
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
random.seed(42)

# Generate sample customer data
n_rows = 1000

data = {
    "customer_id": list(range(1, n_rows + 1)),
    "name": [
        f"Customer {i}" if random.random() > 0.05 else None
        for i in range(n_rows)
    ],
    "email": [
        f"customer{i}@example.com" if random.random() > 0.1 else "invalid-email"
        for i in range(n_rows)
    ],
    "age": [
        random.randint(18, 80) if random.random() > 0.02 else -1
        for _ in range(n_rows)
    ],
    "signup_date": [
        (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        if random.random() > 0.03 else "invalid-date"
        for _ in range(n_rows)
    ],
    "country": random.choices(
        ["US", "UK", "CA", "AU", None],
        weights=[40, 20, 15, 15, 10],
        k=n_rows
    ),
}

# Add some duplicates
data["customer_id"][500:510] = list(range(1, 11))  # Duplicate IDs

df = pl.DataFrame(data)
df.write_csv("customers.csv")

print(f"Created customers.csv with {n_rows} rows")
```

## Step 2: Learn the Schema

Before validating, let Truthound learn the expected schema:

=== "CLI"

    ```bash
    truthound learn customers.csv -o customer_schema.yaml
    ```

    Output:
    ```
    Schema saved to customer_schema.yaml
      Columns: 6
      Rows: 1000
    ```

=== "Python"

    ```python
    from truthound.schema import learn

    schema = learn("customers.csv")
    schema.save("customer_schema.yaml")

    print(f"Learned schema with {len(schema.columns)} columns")
    ```

The schema captures:

- Column names and data types
- Null ratios
- Unique ratios
- Min/max values for numeric columns
- Allowed values for low-cardinality columns (categorical)
- Statistical summaries (mean, std, quantiles)

## Step 3: Validate Your Data

Run validation with all built-in validators:

=== "CLI"

    ```bash
    truthound check customers.csv --schema customer_schema.yaml
    ```

=== "Python"

    ```python
    import truthound as th

    report = th.check(
        "customers.csv",
        schema="customer_schema.yaml"
    )

    # Print formatted summary (uses Rich for pretty output)
    report.print()

    # Or access detailed issues programmatically
    for issue in report.issues:
        detail = f" - {issue.details}" if issue.details else ""
        print(f"[{issue.severity.value}] {issue.column}: {issue.issue_type}{detail}")
    ```

## Step 4: Understand the Results

The validation report shows:

```
Truthound Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Column       ┃ Issue              ┃ Count ┃ Severity ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ customer_id  │ unique_violation   │    10 │ critical │
│ age          │ out_of_range       │    20 │   high   │
│ name         │ null               │    50 │   high   │
│ email        │ invalid_format     │   100 │   high   │
│ signup_date  │ invalid_format     │    30 │  medium  │
│ country      │ null               │   100 │  medium  │
└──────────────┴────────────────────┴───────┴──────────┘

Summary: 6 issues found
```

### Severity Levels

| Severity | Description |
|----------|-------------|
| `critical` | Data integrity issues (duplicates, constraint violations) |
| `high` | Business logic violations (nulls in required fields, out-of-range values) |
| `medium` | Format issues (invalid dates, malformed emails) |
| `low` | Minor issues (whitespace, casing) |

### Issue Types

| Issue Type | Description |
|------------|-------------|
| `unique_violation` | Duplicate values in columns that should be unique |
| `out_of_range` | Values outside expected min/max bounds |
| `null` | Null values in columns |
| `invalid_format` | Values not matching expected format (email, date, etc.) |

## Step 5: Generate Detailed Report

Create an HTML report for stakeholders:

=== "CLI"

    ```bash
    truthound check customers.csv \
        --schema customer_schema.yaml \
        --format html \
        --output report.html
    ```

    !!! note "HTML format requires jinja2"
        Install with: `pip install truthound[reports]` or `pip install jinja2`

=== "Python"

    ```python
    import truthound as th
    from truthound.html_reporter import write_html_report, HTMLReportConfig

    # Run validation
    report = th.check("customers.csv", schema="customer_schema.yaml")

    # Write HTML report to file (option 1: using config)
    config = HTMLReportConfig(title="Customer Data Quality Report")
    write_html_report(report, "report.html", config=config)

    # Or simply (option 2: using kwargs)
    write_html_report(report, "report.html", title="Customer Data Quality Report")

    # Generate HTML string without writing to file
    from truthound.html_reporter import generate_html_report
    html_content = generate_html_report(report, title="Customer Data Quality Report")
    ```

    !!! note "HTML format requires jinja2"
        Install with: `pip install truthound[reports]` or `pip install jinja2`

## Step 6: Fix Issues Programmatically

Use the report to identify and fix issues:

```python
import polars as pl
import truthound as th

# Load data
df = pl.read_csv("customers.csv")

# Fix duplicates - keep first occurrence
df_cleaned = df.unique(subset=["customer_id"], keep="first")

# Fix invalid ages (negative values to null)
df_cleaned = df_cleaned.with_columns(
    pl.when(pl.col("age") < 0)
    .then(None)
    .otherwise(pl.col("age"))
    .alias("age")
)

# Fix invalid emails - mark as null
df_cleaned = df_cleaned.with_columns(
    pl.when(~pl.col("email").str.contains("@"))
    .then(None)
    .otherwise(pl.col("email"))
    .alias("email")
)

# Fix invalid dates - mark as null
df_cleaned = df_cleaned.with_columns(
    pl.when(~pl.col("signup_date").str.contains(r"^\d{4}-\d{2}-\d{2}$"))
    .then(None)
    .otherwise(pl.col("signup_date"))
    .alias("signup_date")
)

# Save cleaned data
df_cleaned.write_csv("customers_cleaned.csv")

# Re-validate
report = th.check("customers_cleaned.csv", schema="customer_schema.yaml")
print(f"Issues remaining: {len(report.issues)}")
```

## Step 7: Set Up Continuous Validation

Create a checkpoint for ongoing validation:

```yaml
# truthound.yaml
checkpoints:
  - name: customer_data_check
    data_source: customers.csv
    validators:
      - "null"
      - "duplicate"
      - "range"
      - "format"
    min_severity: medium
    fail_on_critical: true
    fail_on_high: false
    timeout_seconds: 3600
    tags:
      dataset: customers
      environment: production
    actions:
      - type: store_result
        store_path: ./validation_results
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
```

!!! info "Checkpoint Configuration Options"
    | Field | Description | Default |
    |-------|-------------|---------|
    | `name` | Unique checkpoint identifier | `"default_checkpoint"` |
    | `data_source` | File path or connection string | `""` |
    | `validators` | List of validator names | All validators |
    | `min_severity` | Minimum severity to include | `None` (all severities) |
    | `fail_on_critical` | Fail if critical issues found | `true` |
    | `fail_on_high` | Fail if high severity issues found | `false` |
    | `timeout_seconds` | Max execution time | `3600` |
    | `tags` | Key-value metadata tags | `{}` |

Run validation:

```bash
truthound checkpoint run customer_data_check --config truthound.yaml --strict
```

Or run ad-hoc validation without a config file:

```bash
truthound checkpoint run quick_check \
    --data customers.csv \
    --validators null,duplicate,range \
    --strict \
    --store ./validation_results
```

### Integrating with GitHub Actions

```yaml
# .github/workflows/data-quality.yml
name: Customer Data Quality

on:
  push:
    paths:
      - 'data/customers.csv'
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Truthound
        run: pip install truthound

      - name: Validate Customer Data
        run: |
          truthound checkpoint run customer_data_check \
            --config truthound.yaml \
            --strict \
            --github-summary
```

## Summary

You've learned how to:

1. **Learn schemas** - Automatically infer expected data structure
2. **Run validation** - Check data against 289 built-in validators
3. **Interpret results** - Understand severity levels and issue types
4. **Generate reports** - Create shareable HTML reports
5. **Fix issues** - Programmatically clean data
6. **Automate** - Set up continuous validation with checkpoints

## Next Steps

- [Validators Guide](../guides/validators.md) - Explore all 289 built-in validators
- [CI/CD Integration](../guides/ci-cd.md) - Advanced pipeline setup
- [Custom Validators](../tutorials/custom-validator.md) - Create your own validators
- [Data Sources](../guides/datasources.md) - Connect to databases and cloud warehouses
