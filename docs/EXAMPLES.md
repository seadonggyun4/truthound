# Usage Examples

This document provides comprehensive examples for using Truthound in various data quality validation scenarios.

---

## Table of Contents

1. [Basic Validation](#1-basic-validation)
2. [Schema-Based Validation](#2-schema-based-validation)
3. [Drift Detection](#3-drift-detection)
4. [Anomaly Detection](#4-anomaly-detection)
5. [PII Detection and Masking](#5-pii-detection-and-masking)
6. [Cross-Table Validation](#6-cross-table-validation)
7. [Time Series Validation](#7-time-series-validation)
8. [Privacy Compliance](#8-privacy-compliance)
9. [CI/CD Integration](#9-cicd-integration)
10. [Custom Validators](#10-custom-validators)

---

## 1. Basic Validation

### Simple Data Check

```python
import truthound as th

# Validate a CSV file
report = th.check("data.csv")
print(report)

# Validate a DataFrame
import polars as pl
df = pl.read_parquet("data.parquet")
report = th.check(df)
```

### Filtering by Severity

```python
# Only show medium severity and above
report = th.check(df, min_severity="medium")

# Only show critical issues
report = th.check(df, min_severity="critical")
```

### Selecting Specific Validators

```python
# Run only specific validators
report = th.check(df, validators=["null", "duplicate", "outlier"])

# Run schema-related validators
report = th.check(df, validators=["column_exists", "column_type", "range"])
```

### Strict Mode (CI/CD)

```python
# Exit with error code if any issues found
report = th.check("data.csv", strict=True)

# Combine with severity filter
report = th.check("data.csv", min_severity="high", strict=True)
```

---

## 2. Schema-Based Validation

### Learning Schema from Data

```python
import truthound as th

# Learn schema with constraint inference
schema = th.learn(
    "baseline.csv",
    infer_constraints=True,
    categorical_threshold=20  # Max unique values for categorical
)

# View the schema
print(schema)

# Save to YAML
schema.save("schema.yaml")
```

### Validating Against Schema

```python
# Load schema and validate
report = th.check("new_data.csv", schema="schema.yaml")

# Validate with in-memory schema
schema = th.learn("baseline.csv")
report = th.check("new_data.csv", schema=schema)
```

### Zero-Configuration with Auto Caching

```python
# First run: learns and caches schema
report = th.check("data.csv", auto_schema=True)

# Subsequent runs: uses cached schema
report = th.check("data.csv", auto_schema=True)

# Cache is automatically invalidated when file changes
```

### Manual Schema Definition

```python
from truthound.schema import Schema, ColumnSchema

schema = Schema(columns=[
    ColumnSchema(
        name="id",
        dtype="Int64",
        nullable=False,
        unique=True
    ),
    ColumnSchema(
        name="email",
        dtype="String",
        nullable=False
    ),
    ColumnSchema(
        name="age",
        dtype="Int64",
        min_value=0,
        max_value=150
    ),
    ColumnSchema(
        name="status",
        dtype="String",
        allowed_values=["active", "inactive", "pending"]
    )
])

report = th.check(df, schema=schema)
```

---

## 3. Drift Detection

### Basic Comparison

```python
import truthound as th

# Compare baseline and current data
drift = th.compare("train.csv", "production.csv")
print(drift)

# Check for high drift
if drift.has_high_drift:
    print("Warning: Significant drift detected!")
    for col, result in drift.results.items():
        if result.has_drift:
            print(f"  - {col}: {result.method} = {result.value:.4f}")
```

### Specifying Detection Method

```python
# Kolmogorov-Smirnov test (numeric data)
drift = th.compare(baseline, current, method="ks")

# Chi-square test (categorical data)
drift = th.compare(baseline, current, method="chi2")

# Population Stability Index
drift = th.compare(baseline, current, method="psi")

# Jensen-Shannon divergence
drift = th.compare(baseline, current, method="js")

# Auto-select based on data type
drift = th.compare(baseline, current, method="auto")
```

### Large Dataset Optimization

```python
# Use sampling for faster comparison (92x speedup)
drift = th.compare(
    "historical.parquet",  # 5M rows
    "current.parquet",      # 5M rows
    sample_size=10000
)
```

### Column-Specific Comparison

```python
# Compare only specific columns
drift = th.compare(
    baseline,
    current,
    columns=["feature1", "feature2", "target"]
)
```

### Export for Monitoring

```python
# Export as JSON for dashboards
import json

drift = th.compare(baseline, current)
with open("drift_report.json", "w") as f:
    f.write(drift.to_json())

# Export as dict
drift_dict = drift.to_dict()
```

---

## 4. Anomaly Detection

### Isolation Forest (Multi-dimensional)

```python
from truthound.validators.anomaly import IsolationForestValidator

validator = IsolationForestValidator(
    columns=["feature1", "feature2", "feature3"],
    contamination=0.05,      # Expected proportion of anomalies
    max_anomaly_ratio=0.1,   # Fail if more than 10% anomalies
    n_estimators=100
)
issues = validator.validate(df.lazy())
```

### Mahalanobis Distance (Correlated Features)

```python
from truthound.validators.anomaly import MahalanobisValidator

validator = MahalanobisValidator(
    columns=["x", "y", "z"],
    threshold=3.0  # Chi-square threshold
)
issues = validator.validate(df.lazy())
```

### IQR-Based Detection (Univariate)

```python
from truthound.validators.anomaly import IQRAnomalyValidator

# Standard outliers (k=1.5)
validator = IQRAnomalyValidator(
    column="value",
    k=1.5
)

# Extreme outliers (k=3.0)
validator = IQRAnomalyValidator(
    column="value",
    k=3.0
)
```

### Local Outlier Factor (Clustered Data)

```python
from truthound.validators.anomaly import LOFValidator

validator = LOFValidator(
    columns=["x", "y"],
    n_neighbors=20,
    max_anomaly_ratio=0.05
)
```

### DBSCAN Anomaly Detection

```python
from truthound.validators.anomaly import DBSCANAnomalyValidator

validator = DBSCANAnomalyValidator(
    columns=["feature1", "feature2"],
    eps=0.5,
    min_samples=5,
    max_noise_ratio=0.1
)
```

---

## 5. PII Detection and Masking

### Scanning for PII

```python
import truthound as th

# Scan for personally identifiable information
pii_report = th.scan(df)
print(pii_report)

# View detected PII by column
for finding in pii_report.findings:
    print(f"{finding.column}: {finding.pii_type} ({finding.confidence}%)")
```

### Masking Sensitive Data

```python
# Redact PII (replace with ***)
masked_df = th.mask(df, strategy="redact")

# Hash PII (deterministic anonymization)
masked_df = th.mask(df, strategy="hash")

# Generate fake data
masked_df = th.mask(df, strategy="fake")

# Mask specific columns only
masked_df = th.mask(df, columns=["email", "phone"], strategy="redact")
```

### Export Anonymized Data

```python
# Mask and export
masked_df = th.mask(df, strategy="hash")
masked_df.write_parquet("anonymized.parquet")
masked_df.write_csv("anonymized.csv")
```

---

## 6. Cross-Table Validation

### Row Count Comparison

```python
from truthound.validators.cross_table import CrossTableRowCountValidator

validator = CrossTableRowCountValidator(
    reference_data=orders_df,
    expected_ratio=1.0,  # Same row count
    tolerance=0.01       # 1% tolerance
)
issues = validator.validate(order_items_df.lazy())
```

### Aggregate Comparison

```python
from truthound.validators.cross_table import CrossTableAggregateValidator

validator = CrossTableAggregateValidator(
    reference_data=summary_df,
    column="total_amount",
    reference_column="expected_total",
    aggregation="sum",
    tolerance=0.001
)
```

### Foreign Key Validation

```python
from truthound.validators.referential import ForeignKeyValidator

validator = ForeignKeyValidator(
    column="customer_id",
    reference_data=customers_df,
    reference_column="id"
)
issues = validator.validate(orders_df.lazy())
```

### Composite Foreign Key

```python
from truthound.validators.referential import CompositeForeignKeyValidator

validator = CompositeForeignKeyValidator(
    columns=["store_id", "product_id"],
    reference_data=inventory_df,
    reference_columns=["store_id", "product_id"]
)
```

---

## 7. Time Series Validation

### Gap Detection

```python
from truthound.validators.timeseries import TimeSeriesGapValidator

validator = TimeSeriesGapValidator(
    timestamp_column="timestamp",
    expected_frequency="1h",  # hourly data
    max_gap_ratio=0.01        # Allow 1% gaps
)
issues = validator.validate(df.lazy())
```

### Monotonicity Check

```python
from truthound.validators.timeseries import TimeSeriesMonotonicValidator

validator = TimeSeriesMonotonicValidator(
    timestamp_column="timestamp",
    strict=True  # No duplicates allowed
)
```

### Seasonality Detection

```python
from truthound.validators.timeseries import SeasonalityValidator

validator = SeasonalityValidator(
    value_column="sales",
    timestamp_column="date",
    expected_period=7,  # Weekly seasonality
    min_strength=0.5
)
```

### Trend Validation

```python
from truthound.validators.timeseries import TrendValidator

validator = TrendValidator(
    value_column="revenue",
    timestamp_column="date",
    expected_direction="increasing",
    min_slope=0.0
)
```

---

## 8. Privacy Compliance

### GDPR Compliance Check

```python
from truthound.validators.privacy import GDPRComplianceValidator

validator = GDPRComplianceValidator(
    pii_columns=["email", "phone", "address"],
    require_consent_column="consent_given",
    require_retention_policy=True
)
issues = validator.validate(df.lazy())
```

### CCPA Compliance Check

```python
from truthound.validators.privacy import CCPAComplianceValidator

validator = CCPAComplianceValidator(
    pii_columns=["email", "ssn"],
    require_opt_out_column="do_not_sell"
)
```

### Data Retention Validation

```python
from truthound.validators.privacy import DataRetentionValidator

validator = DataRetentionValidator(
    timestamp_column="created_at",
    max_retention_days=365,
    pii_columns=["email", "phone"]
)
```

---

## 9. CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on:
  push:
    paths:
      - 'data/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Truthound
        run: pip install truthound[all]

      - name: Run validation
        run: truthound check data/input.csv --strict --min-severity medium

      - name: Check for drift
        run: truthound compare data/baseline.csv data/current.csv --format json > drift.json
```

### Python Script for CI

```python
#!/usr/bin/env python3
import sys
import truthound as th

def main():
    # Run validation
    report = th.check("data/input.csv", min_severity="medium")

    # Check for critical issues
    critical_issues = [i for i in report.issues if i.severity == "critical"]

    if critical_issues:
        print(f"Found {len(critical_issues)} critical issues!")
        for issue in critical_issues:
            print(f"  - {issue.column}: {issue.message}")
        sys.exit(1)

    # Check for drift
    drift = th.compare("data/baseline.csv", "data/current.csv")

    if drift.has_high_drift:
        print("Significant drift detected!")
        sys.exit(1)

    print("All checks passed!")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: truthound-check
        name: Truthound Data Validation
        entry: truthound check
        language: system
        files: \.(csv|parquet|json)$
        args: ['--strict', '--min-severity', 'high']
```

---

## 10. Custom Validators

### Creating a Custom Validator

```python
from truthound.validators.base import BaseValidator
from truthound.types import Issue, Severity
import polars as pl

class BusinessRuleValidator(BaseValidator):
    """Validates custom business rules."""

    name = "business_rule"

    def __init__(
        self,
        rule_name: str,
        expression: pl.Expr,
        severity: Severity = "medium",
        mostly: float = 1.0
    ):
        super().__init__(severity=severity, mostly=mostly)
        self.rule_name = rule_name
        self.expression = expression

    def validate(self, lf: pl.LazyFrame) -> list[Issue]:
        # Count violations
        total = lf.select(pl.len()).collect().item()
        violations = lf.filter(~self.expression).select(pl.len()).collect().item()

        pass_rate = 1 - (violations / total) if total > 0 else 1.0

        if pass_rate < self.mostly:
            return [Issue(
                validator=self.name,
                column=None,
                message=f"Business rule '{self.rule_name}' violated: {violations}/{total} rows failed",
                severity=self.severity,
                details={"pass_rate": pass_rate, "violations": violations}
            )]

        return []
```

### Using Custom Validators

```python
import polars as pl

# Define business rule
validator = BusinessRuleValidator(
    rule_name="revenue_positive",
    expression=pl.col("revenue") > 0,
    severity="high",
    mostly=0.99
)

# Validate
issues = validator.validate(df.lazy())
```

### Registering Custom Validators

```python
from truthound.validators.registry import register_validator

@register_validator("my_custom_validator")
class MyCustomValidator(BaseValidator):
    # ... implementation
    pass

# Now usable via th.check()
report = th.check(df, validators=["my_custom_validator"])
```

---

## Command Line Examples

### Basic CLI Usage

```bash
# Validate a file
truthound check data.csv

# With specific validators
truthound check data.csv --validators null,duplicate,outlier

# Filter by severity
truthound check data.csv --min-severity medium

# Strict mode (exit code 1 on issues)
truthound check data.csv --strict

# Output as JSON
truthound check data.csv --format json > report.json
```

### PII Scanning

```bash
# Scan for PII
truthound scan data.csv

# Output as JSON
truthound scan data.csv --format json
```

### Drift Detection

```bash
# Basic comparison
truthound compare baseline.csv current.csv

# With specific method
truthound compare baseline.csv current.csv --method psi

# With sampling for large files
truthound compare train.parquet prod.parquet --sample-size 10000
```

### Profiling

```bash
# Generate data profile
truthound profile data.csv

# Output as JSON
truthound profile data.csv --format json
```

---

[← Back to README](../README.md)
