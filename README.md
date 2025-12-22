<h1 align="center">Truthound</h1>

<p align="center">
  <strong>Zero-Configuration Data Quality Toolkit Powered by Polars</strong>
</p>

<p align="center">
  <em>"Sniffs out bad data. Rust fast."</em>
</p>

---

## Abstract

Truthound is a high-performance data quality validation framework designed for modern data engineering pipelines. The library leverages the computational efficiency of Polars—a Rust-based DataFrame library—to achieve order-of-magnitude performance improvements over traditional Python-based validation solutions. This document presents the architectural design, implemented features, performance benchmarks, and empirical validation results of Truthound v0.1.0.

**Keywords**: Data Quality, Data Validation, Statistical Drift Detection, PII Detection, Polars, Schema Inference

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Statistical Methods](#4-statistical-methods)
5. [Performance Analysis](#5-performance-analysis)
6. [Test Coverage](#6-test-coverage)
7. [API Reference](#7-api-reference)
8. [Comparative Analysis](#8-comparative-analysis)
9. [Installation](#9-installation)
10. [Usage Examples](#10-usage-examples)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [References](#12-references)

---

## 1. Introduction

### 1.1 Problem Statement

Data quality issues represent a significant challenge in modern data engineering workflows. According to industry reports, data scientists spend approximately 60-80% of their time on data preparation and cleaning tasks. Traditional data quality tools often require extensive configuration, suffer from performance limitations when processing large datasets, and lack native support for modern columnar data formats.

### 1.2 Design Goals

Truthound was designed with the following objectives:

1. **Zero Configuration**: Immediate usability without boilerplate setup code
2. **High Performance**: Leveraging Rust-based Polars for computational efficiency
3. **Universal Input Support**: Native handling of diverse data formats
4. **Statistical Rigor**: Implementation of well-established statistical methods for drift detection
5. **Privacy Awareness**: Built-in PII detection and data masking capabilities

### 1.3 Contributions

This work presents:

- A unified data adapter layer supporting multiple input formats
- Optimized validation algorithms using Polars LazyFrame for memory-efficient processing
- Implementation of four statistical drift detection methods
- Automatic schema inference and fingerprint-based caching system
- Comprehensive PII detection patterns including Korean-specific identifiers

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface Layer                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Python API (th.check, th.scan, th.compare)  │  CLI (truthound check)  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Input Adapter Layer                             │
├─────────────────────────────────────────────────────────────────────────┤
│  pandas.DataFrame  │  polars.DataFrame  │  polars.LazyFrame  │  dict   │
│  CSV               │  JSON              │  Parquet           │  Path   │
│                               ↓                                         │
│                    Unified Polars LazyFrame                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Validators    │  │  Drift Detectors │  │   PII Scanners  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • NullValidator │  │ • KS Test       │  │ • Email         │
│ • TypeValidator │  │ • PSI           │  │ • Phone         │
│ • RangeValidator│  │ • Chi-Square    │  │ • SSN           │
│ • OutlierValidator│ │ • Jensen-Shannon│  │ • Credit Card   │
│ • DuplicateValidator│              │  │ • Korean RRN    │
│ • FormatValidator│                  │  │ • Korean Phone  │
│ • UniqueValidator│                  │  │ • Bank Account  │
│ • SchemaValidator│                  │  │ • Passport      │
└────────┬────────┘  └────────┬───────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Schema System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Schema Learning (th.learn)  │  YAML Serialization  │  Fingerprint Cache│
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Output Layer                                    │
├─────────────────────────────────────────────────────────────────────────┤
│      Console (Rich)      │       JSON        │        HTML              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Design Principles

The architecture follows several key design principles:

- **Lazy Evaluation**: All data transformations are performed using Polars LazyFrame to enable query optimization and memory-efficient processing
- **Single Collection Pattern**: Validators are optimized to minimize `collect()` calls, reducing computational overhead
- **Batch Query Optimization**: Statistical computations are batched into single queries where possible

---

## 3. Core Components

### 3.1 Validators

| Validator | Description | Severity Thresholds |
|-----------|-------------|---------------------|
| `NullValidator` | Detects missing/null values per column | >50%: CRITICAL, >20%: HIGH, >5%: MEDIUM, else: LOW |
| `TypeValidator` | Verifies type consistency within columns | Based on inconsistency ratio |
| `RangeValidator` | Identifies out-of-range values | Configurable bounds |
| `OutlierValidator` | Statistical outlier detection using IQR method | >10%: MEDIUM, else: LOW |
| `DuplicateValidator` | Detects duplicate rows | >30%: CRITICAL, >10%: HIGH, >1%: MEDIUM, else: LOW |
| `FormatValidator` | Pattern validation (email, phone, etc.) | Based on invalid ratio |
| `UniqueValidator` | Enforces uniqueness constraints | Column-specific |
| `SchemaValidator` | Validates against learned schema | Type/constraint based |

### 3.2 Drift Detectors

| Detector | Method | Best For | Threshold |
|----------|--------|----------|-----------|
| `KSTestDetector` | Kolmogorov-Smirnov Test | Continuous numeric distributions | p-value < 0.05 |
| `PSIDetector` | Population Stability Index | Model feature monitoring | PSI ≥ 0.1 (moderate), ≥ 0.25 (significant) |
| `ChiSquareDetector` | Chi-Square Test | Categorical distributions | p-value < 0.05 |
| `JensenShannonDetector` | Jensen-Shannon Divergence | Any distribution (symmetric, bounded) | JS ≥ 0.1 |

### 3.3 Schema System

The schema system provides automatic constraint inference:

```python
@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False

    # Constraints (inferred)
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None

    # Statistics (learned)
    null_ratio: float | None = None
    unique_ratio: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] | None = None
```

### 3.4 Auto Schema Caching

The fingerprint-based caching system enables true zero-configuration validation:

1. **Fingerprint Generation**: Combines file path, modification time, and size
2. **Cache Storage**: `.truthound/` directory with JSON index
3. **Invalidation**: Automatic re-learning when data changes

---

## 4. Statistical Methods

### 4.1 Outlier Detection (IQR Method)

The Interquartile Range (IQR) method identifies statistical outliers:

```
IQR = Q3 - Q1
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

Where `k = 1.5` for standard outliers, `k = 3.0` for extreme outliers.

**Implementation Optimization**: Single-pass computation of Q1, Q3, and outlier counts for all numeric columns.

### 4.2 Kolmogorov-Smirnov Test

Measures maximum difference between empirical cumulative distribution functions:

```
D = max|F₁(x) - F₂(x)|
```

P-value approximation uses the asymptotic Kolmogorov distribution.

### 4.3 Population Stability Index (PSI)

Quantifies distribution shift between baseline and current populations:

```
PSI = Σ (Pᵢ - Qᵢ) × ln(Pᵢ / Qᵢ)
```

Where Pᵢ and Qᵢ are proportions in bin i for baseline and current distributions.

**Industry Standard Interpretation**:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.25: Moderate change
- PSI ≥ 0.25: Significant change

### 4.4 Chi-Square Test

Tests independence between observed and expected categorical frequencies:

```
χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
```

P-value computed using Wilson-Hilferty approximation.

### 4.5 Jensen-Shannon Divergence

Symmetric measure of distribution similarity (bounded [0, 1]):

```
JS(P||Q) = ½ KL(P||M) + ½ KL(Q||M)
```

Where M = ½(P + Q) and KL is the Kullback-Leibler divergence.

---

## 5. Performance Analysis

### 5.1 Benchmark Environment

- **Hardware**: Apple Silicon / x86_64
- **Python**: 3.11+
- **Polars**: 1.x

### 5.2 Large-Scale Performance (10M Rows)

| Operation | Time | Throughput |
|-----------|------|------------|
| `th.check()` | 3.53s | 2.83M rows/sec |
| `th.profile()` | 0.15s | 66.7M rows/sec |
| `th.learn()` | 0.27s | 37.0M rows/sec |

### 5.3 Drift Detection Performance

| Dataset Size | Without Sampling | With Sampling (10K) | Speedup |
|--------------|------------------|---------------------|---------|
| 5M vs 5M rows | 3.68s | 0.04s | **92× faster** |

### 5.4 Throughput Testing

Repeated validation operations on 1,000 iterations:

- **Throughput**: 258.7 operations/second
- **Average Latency**: 3.87ms per operation

### 5.5 Memory Efficiency

The LazyFrame-based architecture enables processing of datasets larger than available RAM through:

1. Predicate pushdown
2. Projection pushdown
3. Streaming execution

---

## 6. Test Coverage

### 6.1 Test Summary

| Test Suite | Test Count | Status |
|------------|------------|--------|
| Unit Tests | 39 | ✅ Pass |
| Stress Tests | 53 | ✅ Pass |
| Extreme Stress Tests | 14 | ✅ Pass |
| Integration Tests | 39 | ✅ Pass |
| **Total** | **145** | **✅ All Pass** |

### 6.2 Test Categories

**Stress Tests** (`test_stress.py`):
- Edge cases (empty data, single row/column)
- All Polars data types (Int8-Int64, Float32/64, String, Boolean, Date, Datetime, Duration, Categorical, List, Struct)
- Real-world patterns (high cardinality, sparse data, time series)
- Malicious inputs (SQL injection patterns, XSS, null bytes, Unicode)
- Memory pressure scenarios

**Extreme Stress Tests** (`test_extreme_stress.py`):
- 10M row datasets
- Financial tick data simulation (stock/crypto)
- Mixed type columns
- High duplicate rates
- Wide datasets (100+ columns)
- Concurrent operations

### 6.3 PII Detection Coverage

| PII Type | Pattern | Confidence |
|----------|---------|------------|
| Email | RFC 5322 compliant | 95% |
| US SSN | `XXX-XX-XXXX` | 98% |
| Phone (International) | ITU-T E.164 | 90% |
| Credit Card | Luhn algorithm validated | 85% |
| Korean RRN (주민등록번호) | `XXXXXX-XXXXXXX` | 98% |
| Korean Phone | `0XX-XXXX-XXXX` | 90% |
| Korean Bank Account | Bank-specific formats | 80% |
| Korean Passport | `MXXXXXXXX` | 85% |

---

## 7. API Reference

### 7.1 Primary Functions

```python
import truthound as th

# Data Quality Validation
report = th.check(
    data,                    # Any supported format
    validators=None,         # Optional: list of validator names
    min_severity=None,       # Optional: "low", "medium", "high", "critical"
    schema=None,             # Optional: Schema object or path
    auto_schema=False        # Enable automatic schema caching
)

# PII Scanning
pii_report = th.scan(data)

# Data Masking
masked_df = th.mask(
    data,
    columns=None,            # Optional: specific columns
    strategy="redact"        # "redact", "hash", or "fake"
)

# Statistical Profiling
profile = th.profile(data)

# Schema Learning
schema = th.learn(
    data,
    infer_constraints=True,  # Infer min/max, allowed values
    categorical_threshold=20 # Max unique values for categorical
)
schema.save("schema.yaml")

# Drift Detection
drift = th.compare(
    baseline,                # Reference dataset
    current,                 # Current dataset
    columns=None,            # Optional: specific columns
    method="auto",           # "auto", "ks", "psi", "chi2", "js"
    threshold=None,          # Optional: custom threshold
    sample_size=None         # Optional: for large datasets
)
```

### 7.2 Command Line Interface

```bash
# Validation
truthound check data.csv
truthound check data.csv --validators null,duplicate --min-severity medium
truthound check data.csv --format json --strict

# PII Scanning
truthound scan data.csv

# Profiling
truthound profile data.csv

# Drift Detection
truthound compare baseline.csv current.csv
truthound compare train.parquet prod.parquet --method psi --sample-size 10000
```

---

## 8. Comparative Analysis

### 8.1 Feature Comparison

| Feature | Truthound | Great Expectations | Pandera | Soda Core |
|---------|-----------|-------------------|---------|-----------|
| Zero Configuration | ✅ | ❌ | ❌ | ❌ |
| Polars Native | ✅ | ❌ | ❌ | ❌ |
| LazyFrame Support | ✅ | ❌ | ❌ | ❌ |
| Drift Detection | ✅ | ⚠️ (Plugin) | ❌ | ✅ |
| PII Detection | ✅ | ❌ | ❌ | ✅ |
| Schema Inference | ✅ | ✅ | ✅ | ✅ |
| Auto Caching | ✅ | ❌ | ❌ | ❌ |
| Validator Count | ~10 | 300+ | 50+ | 100+ |

### 8.2 Honest Assessment

**Strengths**:
1. Performance advantage from Polars (not unique to Truthound)
2. True zero-configuration with auto schema caching
3. Statistical drift detection with multiple methods
4. Korean-specific PII patterns

**Limitations** (see Section 11):
1. Limited validator count compared to established solutions
2. No production deployment validation
3. No ecosystem integrations (Airflow, dbt, etc.)
4. Limited documentation and community

---

## 9. Installation

### 9.1 Requirements

- Python 3.11+
- Polars 1.x
- PyYAML
- Rich (for console output)
- Typer (for CLI)

### 9.2 Installation

```bash
pip install truthound
```

### 9.3 Development Setup

```bash
git clone https://github.com/seadonggyun4/Truthound.git
cd Truthound
pip install hatch
hatch env create
hatch run test
```

---

## 10. Usage Examples

### 10.1 Basic Validation

```python
import truthound as th

# Simple validation
report = th.check("data.csv")
print(report)

# With severity filter
report = th.check(df, min_severity="medium")

# Specific validators
report = th.check(df, validators=["null", "duplicate", "outlier"])
```

### 10.2 Schema-Based Validation

```python
# Learn schema from baseline data
schema = th.learn("baseline.csv")
schema.save("schema.yaml")

# Validate new data against schema
report = th.check("new_data.csv", schema="schema.yaml")

# Zero-config with auto caching
report = th.check("data.csv", auto_schema=True)
```

### 10.3 Drift Detection

```python
# Basic comparison
drift = th.compare("train.csv", "production.csv")
print(drift)

if drift.has_high_drift:
    print("⚠️ Significant drift detected!")

# Large dataset with sampling
drift = th.compare(
    "historical.parquet",
    "current.parquet",
    sample_size=10000  # 92× speedup
)

# Export for CI/CD
with open("drift_report.json", "w") as f:
    f.write(drift.to_json())
```

### 10.4 PII Detection

```python
# Scan for PII
pii_report = th.scan(df)

# Mask sensitive data
masked_df = th.mask(df, strategy="hash")
masked_df.write_parquet("anonymized.parquet")
```

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

1. **Validator Coverage**: ~10 validators vs 300+ in Great Expectations
2. **No Production Validation**: Untested in large-scale production environments
3. **Limited Integrations**: No native support for Airflow, dbt, Dagster, etc.
4. **Documentation**: Minimal API documentation and tutorials
5. **Community**: No established user community or support channels

### 11.2 Planned Improvements

1. **Phase 1**: Expand validator library (50+ validators)
2. **Phase 2**: Add pipeline integrations (Airflow, Prefect)
3. **Phase 3**: Web dashboard for visualization
4. **Phase 4**: Database connectors (PostgreSQL, BigQuery)
5. **Phase 5**: Real-time streaming validation

---

## 12. References

1. Polars Documentation. https://pola.rs/
2. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
3. Pearson, K. (1900). "On the criterion that a given system of deviations..."
4. Lin, J. (1991). "Divergence measures based on the Shannon entropy"
5. Great Expectations Documentation. https://greatexpectations.io/
6. Pandera Documentation. https://pandera.readthedocs.io/

---

## License

MIT License

Copyright (c) 2024-2025 Truthound Contributors

---

## Acknowledgments

Built with:
- [Polars](https://pola.rs/) — High-performance DataFrame library
- [Rich](https://rich.readthedocs.io/) — Terminal formatting
- [Typer](https://typer.tiangolo.com/) — CLI framework

---

<p align="center">
  <strong>Truthound — Your data's loyal guardian.</strong>
</p>
