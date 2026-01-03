# First Validation Tutorial

This tutorial walks you through your first data validation with Truthound.

## Objective

By the end of this tutorial, you will:

1. Understand Truthound's validation workflow
2. Create and validate a dataset
3. Interpret validation results
4. Fix data quality issues

## Step 1: Prepare Your Data

Let's create a realistic dataset with common data quality issues:

```python
import polars as pl
from datetime import datetime, timedelta
import random

# Generate sample customer data
n_rows = 1000

data = {
    "customer_id": list(range(1, n_rows + 1)),
    "name": [f"Customer {i}" if random.random() > 0.05 else None for i in range(n_rows)],
    "email": [
        f"customer{i}@example.com" if random.random() > 0.1 else "invalid-email"
        for i in range(n_rows)
    ],
    "age": [random.randint(18, 80) if random.random() > 0.02 else -1 for _ in range(n_rows)],
    "signup_date": [
        (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        if random.random() > 0.03 else "invalid-date"
        for _ in range(n_rows)
    ],
    "country": random.choices(["US", "UK", "CA", "AU", None], weights=[40, 20, 15, 15, 10], k=n_rows),
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

=== "Python"

    ```python
    import truthound as th

    schema = th.learn("customers.csv", infer_constraints=True)
    schema.save("customer_schema.yaml")

    print(f"Learned schema with {len(schema.columns)} columns")
    ```

The schema captures:

- Column names and types
- Null constraints
- Uniqueness constraints
- Value ranges
- Format patterns

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

    # Print summary
    print(report)

    # Access detailed issues
    for issue in report.issues:
        print(f"{issue.severity}: {issue.validator} - {issue.message}")
    ```

## Step 4: Understand the Results

The validation report shows:

```
Data Quality Report
===================
File: customers.csv
Rows: 1,000
Columns: 6

Summary:
  ✓ Passed: 4 validators
  ✗ Failed: 5 validators

Issues (by severity):

  HIGH (2):
    - duplicate_check: customer_id has 10 duplicate values
    - range_check: age contains 20 values outside range [0, 150]

  MEDIUM (3):
    - null_check: name has 50 null values (5.0%)
    - format_check: email has 100 invalid formats (10.0%)
    - date_check: signup_date has 30 invalid dates (3.0%)
```

### Issue Breakdown

| Issue | Column | Count | Impact |
|-------|--------|-------|--------|
| Duplicates | customer_id | 10 | Data integrity |
| Invalid range | age | 20 | Business logic |
| Nulls | name | 50 | Completeness |
| Invalid email | email | 100 | Format |
| Invalid date | signup_date | 30 | Format |

## Step 5: Generate Detailed Report

Create an HTML report for stakeholders:

=== "CLI"

    ```bash
    truthound check customers.csv \
        --schema customer_schema.yaml \
        --format html \
        --output report.html
    ```

=== "Python"

    ```python
    from truthound.datadocs import generate_html_report

    # First, profile the data
    profile = th.profile("customers.csv")

    # Generate HTML report
    generate_html_report(
        profile=profile.to_dict(),
        title="Customer Data Quality Report",
        output_path="report.html"
    )
    ```

## Step 6: Fix Issues Programmatically

Use the report to identify and fix issues:

```python
import polars as pl

# Load data
df = pl.read_csv("customers.csv")

# Fix duplicates - keep first occurrence
df_cleaned = df.unique(subset=["customer_id"], keep="first")

# Fix invalid ages
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

# Save cleaned data
df_cleaned.write_csv("customers_cleaned.csv")

# Re-validate
report = th.check("customers_cleaned.csv")
print(f"Issues remaining: {report.issue_count}")
```

## Step 7: Set Up Continuous Validation

Create a checkpoint for ongoing validation:

```yaml
# truthound.yaml
checkpoints:
  - name: customer_data_check
    data_source: customers.csv
    validators:
      - null
      - duplicate
      - range
      - format
    min_severity: medium
    actions:
      - type: store_result
        store_path: ./validation_results
      - type: slack
        webhook_url: ${SLACK_WEBHOOK}
        notify_on: failure
```

Run scheduled validation:

```bash
truthound checkpoint run customer_data_check --config truthound.yaml --strict
```

## Summary

You've learned how to:

1. **Learn schemas** - Automatically infer expected data structure
2. **Run validation** - Check data against validators
3. **Interpret results** - Understand severity and impact
4. **Generate reports** - Create shareable HTML reports
5. **Fix issues** - Programmatically clean data
6. **Automate** - Set up continuous validation

## Next Steps

- [Validators Guide](../user-guide/validators.md) - Explore 289 built-in validators
- [CI/CD Integration](../user-guide/ci-cd.md) - Set up automated pipelines
- [Custom Validators](../tutorials/custom-validator.md) - Create your own validators
