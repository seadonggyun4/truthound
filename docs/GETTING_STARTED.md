# Getting Started

This guide provides a comprehensive introduction to Truthound, covering installation, basic usage, and common workflows for data quality validation.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Core Concepts](#3-core-concepts)
4. [Basic Validation](#4-basic-validation)
5. [Schema Learning](#5-schema-learning)
6. [Drift Detection](#6-drift-detection)
7. [PII Detection](#7-pii-detection)
8. [Data Profiling](#8-data-profiling)
9. [CLI Usage](#9-cli-usage)
10. [Next Steps](#10-next-steps)

---

## 1. Installation

### Requirements

- Python 3.11 or higher
- Polars 1.x

### Basic Installation

```bash
pip install truthound
```

### Optional Dependencies

Truthound provides optional extras for specific use cases:

```bash
# Statistical drift detection (scipy)
pip install truthound[drift]

# Machine learning-based anomaly detection (scikit-learn)
pip install truthound[anomaly]

# PDF report export (weasyprint)
pip install truthound[pdf]

# Interactive dashboard (reflex)
pip install truthound[dashboard]

# Cloud data warehouse support
pip install truthound[bigquery]    # Google BigQuery
pip install truthound[snowflake]   # Snowflake
pip install truthound[redshift]    # Amazon Redshift
pip install truthound[databricks]  # Databricks

# Enterprise database support
pip install truthound[oracle]      # Oracle Database
pip install truthound[sqlserver]   # Microsoft SQL Server
pip install truthound[enterprise]  # All enterprise sources

# Full installation with all features
pip install truthound[all]
```

### Development Installation

```bash
git clone https://github.com/seadonggyun4/Truthound.git
cd Truthound
pip install hatch
hatch env create
hatch run test
```

---

## 2. Quick Start

### Python API

```python
import truthound as th

# Load and validate data
report = th.check("data.csv")

# Display results
print(report)
```

### CLI

```bash
truthound check data.csv
```

Both approaches produce a validation report identifying data quality issues such as null values, duplicates, type mismatches, and pattern violations.

---

## 3. Core Concepts

### Design Principles

Truthound is built on the following principles:

| Principle | Description |
|-----------|-------------|
| **Zero Configuration** | Immediate usability without boilerplate setup |
| **High Performance** | Polars LazyFrame architecture for memory efficiency |
| **Universal Input** | Support for diverse data formats and sources |
| **Statistical Rigor** | Well-established statistical methods for drift and anomaly detection |
| **Privacy Awareness** | Built-in PII detection and data masking |
| **Extensibility** | Modular architecture for custom validators |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│   Python API (th.check, th.scan, th.compare)  │  CLI        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Input Adapter                           │
│  DataFrame │ CSV │ Parquet │ JSON │ SQL │ Spark │ Cloud DW  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Unified Polars LazyFrame                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Validators  │     │   Drift     │     │    PII      │
   │             │     │  Detectors  │     │  Scanners   │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Layer                            │
│       Console  │  JSON  │  HTML  │  Markdown  │  Stores     │
└─────────────────────────────────────────────────────────────┘
```

### Supported Input Formats

| Format | Example |
|--------|---------|
| Polars DataFrame | `pl.DataFrame({"a": [1, 2, 3]})` |
| Pandas DataFrame | `pd.DataFrame({"a": [1, 2, 3]})` |
| Polars LazyFrame | `pl.scan_csv("data.csv")` |
| CSV file | `"data.csv"` or `Path("data.csv")` |
| Parquet file | `"data.parquet"` |
| JSON file | `"data.json"` |
| Dictionary | `{"col1": [1, 2], "col2": ["a", "b"]}` |

---

## 4. Basic Validation

### Simple Validation

```python
import truthound as th

# Validate any supported input
report = th.check("data.csv")

# Access validation results
print(f"Total issues: {len(report.issues)}")
print(f"Passed: {report.passed}")

# Iterate through issues
for issue in report.issues:
    print(f"[{issue.severity}] {issue.validator}: {issue.message}")
```

### Filtered Validation

```python
# Run specific validators only
report = th.check(
    "data.csv",
    validators=["null", "duplicate", "range"]
)

# Filter by minimum severity
report = th.check(
    "data.csv",
    min_severity="medium"  # "low", "medium", "high", "critical"
)
```

### Validation with Schema

```python
# Load a pre-defined schema
schema = th.Schema.from_yaml("schema.yaml")

# Validate against schema
report = th.check("data.csv", schema=schema)
```

### Auto-Schema Mode

```python
# Enable automatic schema caching
# First run: learns schema and caches it
# Subsequent runs: validates against cached schema
report = th.check("data.csv", auto_schema=True)
```

---

## 5. Schema Learning

Schema learning automatically infers validation constraints from a representative dataset.

### Basic Schema Learning

```python
import truthound as th

# Learn schema from baseline data
schema = th.learn("baseline.csv")

# Display inferred constraints
for col in schema.columns:
    print(f"{col.name}: {col.dtype}")
    if col.min_value is not None:
        print(f"  Range: [{col.min_value}, {col.max_value}]")
    if col.allowed_values is not None:
        print(f"  Allowed values: {col.allowed_values}")
```

### Schema Configuration

```python
schema = th.learn(
    "baseline.csv",
    infer_constraints=True,      # Infer min/max, allowed values
    categorical_threshold=20     # Max unique values for categorical
)
```

### Schema Persistence

```python
# Save schema to file
schema.save("schema.yaml")

# Load schema from file
loaded_schema = th.Schema.from_yaml("schema.yaml")

# Validate new data against saved schema
report = th.check("new_data.csv", schema=loaded_schema)
```

### Schema YAML Format

```yaml
name: customer_data
version: "1.0"
columns:
  - name: id
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
```

---

## 6. Drift Detection

Drift detection identifies distributional changes between baseline and current datasets.

### Basic Drift Detection

```python
import truthound as th

# Compare two datasets
drift_report = th.compare("train.csv", "production.csv")

# Check for drift
if drift_report.has_drift:
    print("Data drift detected!")
    for column, metrics in drift_report.column_results.items():
        if metrics.has_drift:
            print(f"  {column}: {metrics.method} = {metrics.score:.4f}")
```

### Detection Methods

```python
# Automatic method selection based on data type
drift = th.compare("baseline.csv", "current.csv", method="auto")

# Specific statistical methods
drift = th.compare("baseline.csv", "current.csv", method="ks")     # Kolmogorov-Smirnov
drift = th.compare("baseline.csv", "current.csv", method="psi")    # Population Stability Index
drift = th.compare("baseline.csv", "current.csv", method="chi2")   # Chi-square
drift = th.compare("baseline.csv", "current.csv", method="js")     # Jensen-Shannon
```

### Column Selection

```python
# Check specific columns only
drift = th.compare(
    "baseline.csv",
    "current.csv",
    columns=["age", "income", "credit_score"]
)
```

### Sampling for Large Datasets

```python
# Sample for faster processing on large datasets
drift = th.compare(
    "large_baseline.csv",
    "large_current.csv",
    sample_size=10000
)
```

### Available Drift Methods

| Method | Best For | Metric |
|--------|----------|--------|
| `ks` | Continuous numeric | D-statistic, p-value |
| `chi2` | Categorical | χ²-statistic, p-value |
| `psi` | Model monitoring | PSI score |
| `js` | Any distribution | JS divergence (0-1) |
| `wasserstein` | Distribution shape | Earth Mover's Distance |
| `kl` | Information loss | KL divergence |

---

## 7. PII Detection

PII (Personally Identifiable Information) detection scans data for sensitive information patterns.

### Basic PII Scanning

```python
import truthound as th

# Scan for PII
pii_report = th.scan("customer_data.csv")

# Display findings
for finding in pii_report.findings:
    print(f"{finding.column}: {finding.pii_type} ({finding.confidence:.0%})")
```

### Detected PII Types

| Type | Pattern |
|------|---------|
| Email | RFC 5322 email format |
| Phone | International phone numbers |
| SSN | US Social Security Numbers |
| Credit Card | Major card formats (Visa, MC, Amex) |
| Korean RRN | Korean Resident Registration Number |
| Korean Phone | Korean phone formats |
| Bank Account | Bank account patterns |
| Passport | Passport number formats |

### Data Masking

```python
# Mask detected PII
masked_df = th.mask(df)

# Specific masking strategies
masked_df = th.mask(df, strategy="redact")   # Replace with ***
masked_df = th.mask(df, strategy="hash")     # SHA-256 hash
masked_df = th.mask(df, strategy="fake")     # Realistic fake data

# Mask specific columns only
masked_df = th.mask(df, columns=["email", "phone", "ssn"])

# Strict mode - fail if column doesn't exist
masked_df = th.mask(df, columns=["email"], strict=True)

# Default behavior - warn and skip missing columns
masked_df = th.mask(df, columns=["email", "nonexistent"])  # Warning: Column 'nonexistent' not found
```

### Masking Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| `redact` | Replace with asterisks | `john@example.com` → `***` |
| `hash` | SHA-256 hash | `john@example.com` → `a8b9c0d1...` |
| `fake` | Realistic fake data | `john@example.com` → `alice@fake.net` |

---

## 8. Data Profiling

Data profiling provides statistical summaries and pattern detection for datasets.

### Basic Profiling

```python
import truthound as th

# Profile a dataset
profile = th.profile("data.csv")

# Display summary
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")

# Column-level statistics
for col in profile.columns:
    print(f"\n{col.name} ({col.dtype}):")
    print(f"  Null ratio: {col.null_ratio:.2%}")
    print(f"  Unique ratio: {col.unique_ratio:.2%}")
    if col.mean is not None:
        print(f"  Mean: {col.mean:.2f}, Std: {col.std:.2f}")
```

### Profile Statistics

| Statistic | Description |
|-----------|-------------|
| `null_ratio` | Proportion of null values |
| `unique_ratio` | Proportion of unique values |
| `mean`, `std` | Central tendency and spread (numeric) |
| `min`, `max` | Value range (numeric) |
| `quantiles` | 25th, 50th, 75th percentiles |
| `patterns` | Detected data patterns |

### HTML Report Generation

```python
from truthound.datadocs import generate_html_report

# Generate HTML report from profile
html = generate_html_report(
    profile=profile.to_dict(),
    title="Data Quality Report",
    theme="professional",
    output_path="report.html"
)
```

---

## 9. CLI Usage

### Validation Commands

```bash
# Basic validation
truthound check data.csv

# Validation with specific validators
truthound check data.csv --validators null,duplicate,range

# Validation with severity filter
truthound check data.csv --min-severity medium

# Strict mode (exit code 1 on failures)
truthound check data.csv --strict

# JSON output
truthound check data.csv --format json
```

### Schema Commands

```bash
# Learn schema from data
truthound learn baseline.csv -o schema.yaml

# Validate against schema
truthound check data.csv --schema schema.yaml
```

### Drift Detection

```bash
# Basic drift detection
truthound compare baseline.csv current.csv

# Specific method
truthound compare baseline.csv current.csv --method psi

# With sampling
truthound compare baseline.csv current.csv --sample-size 10000
```

### PII Scanning

```bash
# Scan for PII
truthound scan data.csv
```

### Profiling and Reports

```bash
# Generate profile
truthound auto-profile data.csv -o profile.json

# Generate validation suite from profile
truthound generate-suite profile.json -o rules.yaml

# One-step profile and suite generation
truthound quick-suite data.csv -o rules.yaml

# Generate HTML report
truthound docs generate profile.json -o report.html --theme professional

# List available options
truthound list-formats      # Export formats
truthound list-presets      # Configuration presets
truthound list-categories   # Rule categories
truthound docs themes       # Available themes
```

### CLI Help

```bash
truthound --help
truthound check --help
truthound compare --help
```

---

## 10. Next Steps

### Explore Documentation

| Topic | Documentation |
|-------|---------------|
| All validators | [Validators Reference](VALIDATORS.md) |
| Statistical methods | [Statistical Methods](STATISTICAL_METHODS.md) |
| Data sources | [Data Sources](DATASOURCES.md) |
| CI/CD integration | [Checkpoint & CI/CD](CHECKPOINT.md) |
| Auto-profiling | [Auto-Profiling](PROFILER.md) |
| HTML reports | [Data Docs](DATADOCS.md) |
| Plugins | [Plugin Architecture](PLUGINS.md) |
| Advanced features | [ML, Lineage, Realtime](ADVANCED.md) |

### Example Workflows

See [Examples](EXAMPLES.md) for comprehensive usage patterns including:

- Schema-based validation pipelines
- Cross-table validation
- Time series validation
- Privacy compliance checks
- CI/CD integration
- Custom validator development

### API Reference

See [API Reference](API_REFERENCE.md) for complete function signatures and configuration options.
