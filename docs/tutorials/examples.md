# Usage Examples

Comprehensive examples for Truthound data quality validation scenarios.

!!! tip "Related Tutorials"
    For step-by-step guidance, see:

    - [Data Profiling Tutorial](data-profiling.md) - Profile data and generate rules
    - [Custom Validator Tutorial](custom-validator.md) - Create custom validators
    - [Enterprise Setup Tutorial](enterprise-setup.md) - CI/CD and production setup

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
9. [Advanced Patterns](#9-advanced-patterns)
10. [CLI Quick Reference](#10-cli-quick-reference)

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
if not report.has_issues:
    print("All validations passed!")
else:
    print(f"Found {len(report.issues)} issues")
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.column}: {issue.issue_type}")
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

# Validate specific columns only
report = th.check(df, columns=["email", "phone", "ssn"])
```

### Strict Mode (CI/CD)

```python
import sys

# Run validation with severity filter
report = th.check("data.csv", min_severity="high")

# Exit with error code if any issues found
if report.has_issues:
    sys.exit(1)

# For critical issues check
if report.has_critical:
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
    categorical_threshold=20
)

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

### Manual Schema Definition

```python
from truthound.schema import Schema, ColumnSchema

# Schema.columns is a dict mapping column name to ColumnSchema
schema = Schema(columns={
    "id": ColumnSchema(
        name="id",
        dtype="Int64",
        nullable=False,
        unique=True
    ),
    "email": ColumnSchema(
        name="email",
        dtype="String",
        nullable=False,
        pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$"  # Regex pattern
    ),
    "age": ColumnSchema(
        name="age",
        dtype="Int64",
        min_value=0,
        max_value=150
    ),
    "status": ColumnSchema(
        name="status",
        dtype="String",
        allowed_values=["active", "inactive", "pending"]
    ),
})

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

if drift.has_drift:
    print("Data drift detected!")
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"  - {col_drift.column}: {col_drift.result.method} = {col_drift.result.statistic:.4f}")

# Check for high drift
if drift.has_high_drift:
    print("WARNING: High drift detected!")

# Get list of drifted column names
drifted_cols = drift.get_drifted_columns()
print(f"Drifted columns: {drifted_cols}")
```

### Specifying Detection Method

```python
import truthound as th

# Auto-select based on data type (default, recommended)
drift = th.compare(baseline, current, method="auto")

# Kolmogorov-Smirnov test (numeric columns only)
drift = th.compare(baseline, current, method="ks")

# Population Stability Index (numeric columns only)
drift = th.compare(baseline, current, method="psi")

# Chi-square test (categorical columns)
drift = th.compare(baseline, current, method="chi2")

# Jensen-Shannon divergence (works with any column type)
drift = th.compare(baseline, current, method="js")

# Custom threshold
drift = th.compare(baseline, current, threshold=0.2)
```

> **Note:** `ks` and `psi` methods only work with numeric columns. Use `columns` parameter
> to specify numeric columns, or use `method="auto"` for automatic method selection.
>
> ```python
> # Compare only numeric columns with PSI
> drift = th.compare(baseline, current, method="psi", columns=["age", "salary"])
> ```

### Large Dataset Optimization

```python
import truthound as th

# Use sampling for faster comparison
drift = th.compare(
    "historical.parquet",
    "current.parquet",
    sample_size=10000,
)
```

---

## 4. Anomaly Detection

### Isolation Forest (Multi-dimensional)

```python
from truthound.validators.anomaly import IsolationForestValidator

validator = IsolationForestValidator(
    columns=["feature1", "feature2", "feature3"],
    contamination=0.05,
    max_anomaly_ratio=0.1,
    n_estimators=100
)
issues = validator.validate(df.lazy())
```

### IQR-Based Detection (Univariate)

```python
from truthound.validators.anomaly import IQRAnomalyValidator

# Standard outliers (iqr_multiplier=1.5)
validator = IQRAnomalyValidator(column="value", iqr_multiplier=1.5)

# Extreme outliers (iqr_multiplier=3.0)
validator = IQRAnomalyValidator(column="value", iqr_multiplier=3.0)
```

### Mahalanobis Distance (Correlated Features)

```python
from truthound.validators.anomaly import MahalanobisValidator

validator = MahalanobisValidator(
    columns=["x", "y", "z"],
    threshold=3.0
)
issues = validator.validate(df.lazy())
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

# View detected PII by column (findings are dicts)
for finding in pii_report.findings:
    print(f"{finding['column']}:")
    print(f"  Type: {finding['pii_type']}")
    print(f"  Confidence: {finding['confidence']}%")
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
masked_df = th.mask(df, columns=["email"], strict=True)
```

---

## 6. Cross-Table Validation

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

### Row Count Comparison

```python
from truthound.validators.cross_table import CrossTableRowCountValidator

validator = CrossTableRowCountValidator(
    reference_data=orders_df,
    reference_name="orders",
    tolerance=0,  # Allow 0 difference
)
issues = validator.validate(order_items_df.lazy())
```

---

## 7. Time Series Validation

### Gap Detection

```python
from truthound.validators.timeseries import TimeSeriesGapValidator, TimeFrequency

validator = TimeSeriesGapValidator(
    timestamp_column="timestamp",
    frequency=TimeFrequency.HOURLY,  # or "hourly"
    max_gap_ratio=0.01
)
issues = validator.validate(df.lazy())
```

### Monotonicity Check

```python
from truthound.validators.timeseries import TimeSeriesMonotonicValidator, MonotonicityType

validator = TimeSeriesMonotonicValidator(
    timestamp_column="timestamp",
    value_column="cumulative_count",
    monotonicity=MonotonicityType.STRICTLY_INCREASING,  # or "strictly_increasing"
)
```

### Seasonality Detection

```python
from truthound.validators.timeseries import SeasonalityValidator

validator = SeasonalityValidator(
    timestamp_column="date",
    value_column="sales",
    expected_period=7,
    min_seasonality_strength=0.3,
)
```

### Trend Validation

```python
from truthound.validators.timeseries import TrendValidator, TrendDirection

validator = TrendValidator(
    timestamp_column="date",
    value_column="revenue",
    expected_direction=TrendDirection.INCREASING,  # or "increasing"
    min_trend_strength=0.01,
)
```

---

## 8. Privacy Compliance

### GDPR Compliance Check

```python
from truthound.validators.privacy import GDPRComplianceValidator

# Auto-detect PII columns based on GDPR definitions
validator = GDPRComplianceValidator(
    columns=["email", "phone", "address"],  # Optional: specific columns
    sample_size=1000,
    min_confidence=70,
    detect_special_categories=True,
)
issues = validator.validate(df.lazy())
```

### CCPA Compliance Check

```python
from truthound.validators.privacy import CCPAComplianceValidator

# Auto-detect California PI based on CCPA definitions
validator = CCPAComplianceValidator(
    columns=["email", "ssn"],  # Optional: specific columns
    sample_size=1000,
)
```

### Data Retention Validation

```python
from truthound.validators.privacy import DataRetentionValidator

validator = DataRetentionValidator(
    date_column="created_at",
    retention_days=365,
    pii_columns=["email", "phone"],
)
```

---

## 9. Advanced Patterns

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

    def run(self, data_path: str) -> dict:
        results = {}

        # Schema validation
        schema = th.Schema.load(self.schema_path)
        results["validation"] = th.check(data_path, schema=schema)

        # Drift detection
        if self.drift_baseline:
            results["drift"] = th.compare(self.drift_baseline, data_path)

        # PII scan
        if self.pii_check:
            results["pii"] = th.scan(data_path)

        # Aggregate status
        has_validation_issues = results["validation"].has_issues
        has_drift = results.get("drift") and results["drift"].has_drift
        has_pii = results.get("pii") and len(results["pii"].findings) > 0

        results["passed"] = not any([has_validation_issues, has_drift, has_pii])

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
    return str(path), not report.has_issues, report.issues

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
        schema = th.Schema.load(schema_path)
        results = {}

        for env, path in self.environments.items():
            report = th.check(path, schema=schema)
            results[env] = {
                "passed": not report.has_issues,
                "issue_count": len(report.issues),
            }

        return results

    def compare_environments(self, baseline_env: str, target_env: str) -> dict:
        """Compare two environments for drift."""
        baseline = self.environments[baseline_env]
        target = self.environments[target_env]
        drift = th.compare(baseline, target)

        return {
            "has_drift": drift.has_drift,
            "has_high_drift": drift.has_high_drift,
            "drifted_columns": drift.get_drifted_columns(),
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

## 10. CLI Quick Reference

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

# With custom threshold
truthound compare baseline.csv current.csv --threshold 0.2

# Strict mode - exit with error if drift found
truthound compare baseline.csv current.csv --strict
```

### Profiling

```bash
# Generate profile
truthound profile data.csv

# Auto-profile with pattern detection
truthound auto-profile data.csv -o profile.json

# Generate validation suite
truthound generate-suite profile.json -o rules.yaml

# One-step profile and suite
truthound quick-suite data.csv -o rules.yaml
```

### Schema Management

```bash
# Learn schema from data
truthound learn data.csv -o schema.yaml

# Validate against schema
truthound check data.csv --schema schema.yaml
```

---

## See Also

- [Getting Started](../getting-started/index.md) - Installation and first steps
- [Validators Guide](../guides/validators.md) - Complete validator documentation
- [Statistical Methods](../concepts/statistical-methods.md) - Drift and anomaly methods
- [Python API Reference](../python-api/index.md) - Complete API documentation
