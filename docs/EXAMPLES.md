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
11. [Data Profiling](#11-data-profiling)
12. [Advanced Patterns](#12-advanced-patterns)
13. [CLI Reference](#13-cli-reference)

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

# Check validation status
if report.passed:
    print("All validations passed!")
else:
    print(f"Found {len(report.issues)} issues")
    for issue in report.issues:
        print(f"  [{issue.severity}] {issue.column}: {issue.message}")
```

### Filtering by Severity

```python
# Only show medium severity and above
report = th.check(df, min_severity="medium")

# Only show high severity and above
report = th.check(df, min_severity="high")

# Only show critical issues
report = th.check(df, min_severity="critical")
```

### Selecting Specific Validators

```python
# Run only specific validators
report = th.check(df, validators=["null", "duplicate", "outlier"])

# Run schema-related validators
report = th.check(df, validators=["column_exists", "column_type", "range"])

# Validate specific columns only
report = th.check(df, columns=["email", "phone", "ssn"])
```

### Strict Mode (CI/CD)

```python
import sys

# Exit with error code if any issues found
report = th.check("data.csv", strict=True)

# Combine with severity filter
report = th.check("data.csv", min_severity="high", strict=True)

# In CI/CD pipelines
if not report.passed:
    sys.exit(1)
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

# View inferred constraints
for col in schema.columns:
    print(f"{col.name}:")
    print(f"  Type: {col.dtype}")
    print(f"  Nullable: {col.nullable}")
    if col.min_value is not None:
        print(f"  Range: [{col.min_value}, {col.max_value}]")
    if col.allowed_values:
        print(f"  Allowed: {col.allowed_values}")

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

# Load from YAML
loaded_schema = th.Schema.from_yaml("schema.yaml")
report = th.check("new_data.csv", schema=loaded_schema)
```

### Zero-Configuration with Auto Caching

```python
# First run: learns and caches schema
report = th.check("data.csv", auto_schema=True)

# Subsequent runs: uses cached schema
report = th.check("data.csv", auto_schema=True)

# Cache is automatically invalidated when file changes
# Schema is cached by data fingerprint
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
        nullable=False,
        patterns=["email"]
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
    ),
    ColumnSchema(
        name="created_at",
        dtype="Datetime",
        nullable=False
    )
])

report = th.check(df, schema=schema)
```

### YAML Schema Format

```yaml
# schema.yaml
name: customer_data
version: "1.0"
columns:
  - name: customer_id
    dtype: Int64
    nullable: false
    unique: true

  - name: email
    dtype: String
    nullable: true
    patterns:
      - email

  - name: age
    dtype: Int64
    nullable: true
    min_value: 0
    max_value: 150

  - name: status
    dtype: String
    nullable: false
    allowed_values:
      - active
      - inactive
      - pending

  - name: created_at
    dtype: Datetime
    nullable: false
```

---

## 3. Drift Detection

### Basic Comparison

```python
import truthound as th

# Compare baseline and current data
drift = th.compare("train.csv", "production.csv")
print(drift)

# Check for drift
if drift.has_drift:
    print("Data drift detected!")

# Check for high drift
if drift.has_high_drift:
    print("Warning: Significant drift detected!")
    for col, result in drift.column_results.items():
        if result.has_drift:
            print(f"  - {col}: {result.method} = {result.score:.4f}")
```

### Specifying Detection Method

```python
# Kolmogorov-Smirnov test (continuous numeric data)
drift = th.compare(baseline, current, method="ks")

# Chi-square test (categorical data)
drift = th.compare(baseline, current, method="chi2")

# Population Stability Index (model monitoring)
drift = th.compare(baseline, current, method="psi")

# Jensen-Shannon divergence (any distribution)
drift = th.compare(baseline, current, method="js")

# Wasserstein distance (distribution shape)
drift = th.compare(baseline, current, method="wasserstein")

# Auto-select based on data type
drift = th.compare(baseline, current, method="auto")
```

### Large Dataset Optimization

```python
# Use sampling for faster comparison (92x speedup)
drift = th.compare(
    "historical.parquet",  # 5M rows
    "current.parquet",      # 5M rows
    sample_size=10000,
    sample_seed=42  # Reproducible sampling
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

# Exclude columns
drift = th.compare(
    baseline,
    current,
    exclude_columns=["id", "timestamp"]
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

### Model Monitoring Pipeline

```python
import truthound as th

def monitor_model_data(train_path: str, prod_path: str) -> dict:
    """Monitor production data for drift from training data."""

    # Detect drift
    drift = th.compare(train_path, prod_path, method="psi")

    # Categorize drift severity
    alerts = {
        "critical": [],
        "warning": [],
        "ok": []
    }

    for col, result in drift.column_results.items():
        if result.score >= 0.25:
            alerts["critical"].append((col, result.score))
        elif result.score >= 0.1:
            alerts["warning"].append((col, result.score))
        else:
            alerts["ok"].append((col, result.score))

    return alerts

# Usage
alerts = monitor_model_data("train.csv", "production.csv")
if alerts["critical"]:
    print("CRITICAL: Model retraining recommended")
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

### Z-Score Detection

```python
from truthound.ml import ZScoreAnomalyDetector

detector = ZScoreAnomalyDetector(threshold=3.0)
detector.fit(df)
result = detector.detect(df)

# View anomalies
anomalies = result.get_anomaly_indices()
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

### Ensemble Anomaly Detection

```python
from truthound.ml import EnsembleAnomalyDetector, ZScoreAnomalyDetector, IQRAnomalyDetector

ensemble = EnsembleAnomalyDetector(
    detectors=[
        ZScoreAnomalyDetector(threshold=3.0),
        IQRAnomalyDetector(multiplier=1.5)
    ],
    voting_strategy="majority"  # or "unanimous", "any"
)
ensemble.fit(df)
result = ensemble.detect(df)
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
    print(f"{finding.column}:")
    print(f"  Type: {finding.pii_type}")
    print(f"  Confidence: {finding.confidence:.0%}")
    print(f"  Sample matches: {finding.sample_count}")
```

### Masking Sensitive Data

```python
import polars as pl
import truthound as th

df = pl.DataFrame({
    "name": ["John Doe", "Jane Smith"],
    "email": ["john@example.com", "jane@example.com"],
    "ssn": ["123-45-6789", "987-65-4321"],
    "phone": ["555-123-4567", "555-987-6543"]
})

# Redact PII (replace with ***)
masked_df = th.mask(df, strategy="redact")

# Hash PII (deterministic anonymization)
masked_df = th.mask(df, strategy="hash")

# Generate fake data
masked_df = th.mask(df, strategy="fake")

# Mask specific columns only
masked_df = th.mask(df, columns=["email", "phone"], strategy="redact")

# Strict mode - fail if column doesn't exist
masked_df = th.mask(df, columns=["email"], strict=True)  # ValueError if missing

# Default mode - warn and skip missing columns
masked_df = th.mask(df, columns=["email", "missing_col"])  # Warning, skips missing
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
import polars as pl

# Parent table
customers = pl.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Carol", "Diana", "Eve"]
})

# Child table
orders = pl.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 6, 3],  # 6 is orphan
    "amount": [100, 200, 150, 300]
})

validator = ForeignKeyValidator(
    column="customer_id",
    reference_data=customers,
    reference_column="customer_id"
)
issues = validator.validate(orders.lazy())
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

### Orphan Record Detection

```python
from truthound.validators.referential import OrphanRecordValidator

validator = OrphanRecordValidator(
    column="order_id",
    reference_data=order_items,
    reference_column="order_id"
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

### Temporal Ordering

```python
from truthound.validators.timeseries import TemporalOrderValidator

validator = TemporalOrderValidator(
    datetime_column="event_time",
    sequence_column="sequence_id",
    order="ascending"
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

### Multi-Regulation Check

```python
import truthound as th

# Check multiple regulations
report = th.scan(df, regulations=["gdpr", "ccpa", "lgpd"])

# Check for compliance
if report.has_violations:
    print("Privacy violations detected:")
    for violation in report.violations:
        print(f"  {violation.regulation}: {violation.column} - {violation.message}")
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
  pull_request:
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
"""Data quality check for CI/CD pipelines."""
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

### Checkpoint Integration

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig

# Configure checkpoint
config = CheckpointConfig(
    store_backend="s3",
    store_config={"bucket": "validation-results"},
    fail_on_error=True,
    notifications=["slack", "email"]
)

# Create checkpoint
checkpoint = Checkpoint("production_validation", config)

# Run validation with checkpointing
with checkpoint:
    report = th.check("data.csv")
    checkpoint.record(report)

# Results are stored and notifications sent automatically
```

---

## 10. Custom Validators

### Creating a Custom Validator

```python
from truthound.validators.base import Validator, ValidationIssue, Severity
import polars as pl

class BusinessRuleValidator(Validator):
    """Validates custom business rules."""

    name = "business_rule"
    category = "custom"

    def __init__(
        self,
        rule_name: str,
        expression: pl.Expr,
        severity: Severity = Severity.MEDIUM,
        mostly: float = 1.0
    ):
        self.rule_name = rule_name
        self.expression = expression
        self.severity = severity
        self.mostly = mostly

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Count violations
        total = lf.select(pl.len()).collect().item()
        violations = lf.filter(~self.expression).select(pl.len()).collect().item()

        pass_rate = 1 - (violations / total) if total > 0 else 1.0

        if pass_rate < self.mostly:
            return [ValidationIssue(
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
import truthound as th

# Define business rule
validator = BusinessRuleValidator(
    rule_name="revenue_positive",
    expression=pl.col("revenue") > 0,
    severity=Severity.HIGH,
    mostly=0.99
)

# Validate
issues = validator.validate(df.lazy())

# Or use with th.check()
report = th.check(df, validators=[validator])
```

### Registering Custom Validators

```python
from truthound.validators.base import register_validator

@register_validator("my_custom_validator")
class MyCustomValidator(Validator):
    name = "my_custom_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Implementation
        ...

# Now usable via th.check()
report = th.check(df, validators=["my_custom_validator"])
```

### Parametric Validator

```python
from dataclasses import dataclass

@dataclass
class RangeConfig:
    min_value: float | None = None
    max_value: float | None = None
    inclusive: bool = True

class ConfigurableRangeValidator(Validator):
    """Validate values within configurable range."""

    name = "configurable_range"
    category = "custom"

    def __init__(self, column: str, config: RangeConfig):
        self.column = column
        self.config = config

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Implementation
        ...
```

---

## 11. Data Profiling

### Basic Profiling

```python
import truthound as th

# Generate profile
profile = th.profile("data.csv")

# Summary statistics
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
print(f"Memory: {profile.memory_usage_mb:.2f} MB")

# Column details
for col in profile.columns:
    print(f"\n{col.name} ({col.dtype}):")
    print(f"  Null ratio: {col.null_ratio:.2%}")
    print(f"  Unique ratio: {col.unique_ratio:.2%}")
    if col.mean is not None:
        print(f"  Mean: {col.mean:.2f}")
        print(f"  Std: {col.std:.2f}")
        print(f"  Range: [{col.min}, {col.max}]")
```

### Generating Validation Rules

```python
from truthound.profiler import AutoProfiler

# Create profiler
profiler = AutoProfiler()

# Profile and generate rules
profile = profiler.profile("data.csv")
rules = profiler.generate_rules(profile)

# Save rules
rules.save("generated_rules.yaml")

# Use rules for validation
report = th.check("new_data.csv", rules=rules)
```

### HTML Report Generation

```python
from truthound.datadocs import generate_html_report

# Generate HTML report
profile = th.profile("data.csv")
html = generate_html_report(
    profile=profile.to_dict(),
    title="Data Quality Report",
    theme="professional",
    output_path="report.html"
)
```

---

## 12. Advanced Patterns

### Validation Pipeline

```python
import truthound as th
from dataclasses import dataclass

@dataclass
class ValidationPipeline:
    """Reusable validation pipeline."""

    schema_path: str
    drift_baseline: str | None = None
    pii_check: bool = True
    strict: bool = False

    def run(self, data_path: str) -> dict:
        results = {}

        # Schema validation
        schema = th.Schema.from_yaml(self.schema_path)
        results["validation"] = th.check(data_path, schema=schema)

        # Drift detection
        if self.drift_baseline:
            results["drift"] = th.compare(self.drift_baseline, data_path)

        # PII scan
        if self.pii_check:
            results["pii"] = th.scan(data_path)

        # Aggregate status
        results["passed"] = all([
            results["validation"].passed,
            not results.get("drift", type("", (), {"has_drift": False})).has_drift,
            not results.get("pii", type("", (), {"has_pii": False})).has_pii
        ])

        return results

# Usage
pipeline = ValidationPipeline(
    schema_path="schema.yaml",
    drift_baseline="baseline.csv",
    pii_check=True
)
results = pipeline.run("new_data.csv")
```

### Batch Validation

```python
import truthound as th
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def validate_file(path: Path) -> tuple[str, bool, list]:
    """Validate a single file."""
    report = th.check(str(path))
    return str(path), report.passed, report.issues

def batch_validate(directory: str, pattern: str = "*.csv") -> dict:
    """Validate all matching files in directory."""
    files = list(Path(directory).glob(pattern))

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(validate_file, f) for f in files]
        for future in futures:
            path, passed, issues = future.result()
            results[path] = {"passed": passed, "issues": issues}

    return results

# Usage
results = batch_validate("data/", "*.csv")
for path, result in results.items():
    status = "PASS" if result["passed"] else "FAIL"
    print(f"{status}: {path} ({len(result['issues'])} issues)")
```

### Multi-Environment Validation

```python
import truthound as th

class EnvironmentValidator:
    """Validate data across multiple environments."""

    def __init__(self, environments: dict[str, str]):
        self.environments = environments

    def validate_all(self, schema_path: str) -> dict:
        """Validate all environments against schema."""
        schema = th.Schema.from_yaml(schema_path)
        results = {}

        for env, path in self.environments.items():
            report = th.check(path, schema=schema)
            results[env] = {
                "passed": report.passed,
                "issue_count": len(report.issues),
                "issues": report.issues
            }

        return results

    def compare_environments(self, baseline_env: str, target_env: str) -> dict:
        """Compare two environments for drift."""
        baseline = self.environments[baseline_env]
        target = self.environments[target_env]

        drift = th.compare(baseline, target)

        return {
            "has_drift": drift.has_drift,
            "drifted_columns": [
                col for col, result in drift.column_results.items()
                if result.has_drift
            ]
        }

# Usage
validator = EnvironmentValidator({
    "dev": "data/dev/customers.csv",
    "staging": "data/staging/customers.csv",
    "prod": "data/prod/customers.csv"
})

results = validator.validate_all("schema.yaml")
drift = validator.compare_environments("staging", "prod")
```

---

## 13. CLI Reference

### Basic Commands

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

### Profiling and Reports

```bash
# Generate data profile
truthound auto-profile data.csv -o profile.json

# Generate validation suite from profile
truthound generate-suite profile.json -o rules.yaml

# One-step profile and suite generation
truthound quick-suite data.csv -o rules.yaml

# With specific categories
truthound quick-suite data.csv -o rules.yaml --categories completeness,uniqueness,range

# Generate HTML report
truthound docs generate profile.json -o report.html --theme professional

# List available options
truthound list-formats      # Export formats
truthound list-presets      # Configuration presets
truthound list-categories   # Rule categories
truthound docs themes       # Available themes
```

---

## See Also

- [Getting Started](GETTING_STARTED.md) — Quick start guide
- [Validators Reference](VALIDATORS.md) — Complete validator documentation
- [Statistical Methods](STATISTICAL_METHODS.md) — Drift and anomaly methods
- [API Reference](API_REFERENCE.md) — Complete API documentation
- [Plugin Architecture](PLUGINS.md) — Creating custom plugins
