<div align="center">
  <img width="500" height="500" alt="logo" src="https://github.com/user-attachments/assets/4b4dea6c-46b9-49e5-af19-744a3b216bf8" />
</div>

<h1 align="center">Truthound</h1>

<p align="center">
  <strong>Zero-Configuration Data Quality Toolkit Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/v/truthound" alt="PyPI"></a>
  <a href="https://pypi.org/project/truthound/"><img src="https://img.shields.io/pypi/pyversions/truthound" alt="Python"></a>
  <a href="https://github.com/seadonggyun4/Truthound/blob/main/LICENSE"><img src="https://img.shields.io/github/license/seadonggyun4/Truthound" alt="License"></a>
</p>

---

## Abstract

Truthound is a high-performance data quality validation framework designed for modern data engineering pipelines. The library leverages the computational efficiency of Polars—a Rust-based DataFrame library—to achieve order-of-magnitude performance improvements over traditional Python-based validation solutions. This document presents the architectural design, implemented features, performance benchmarks, and empirical validation results of Truthound.

**Keywords**: Data Quality, Data Validation, Statistical Drift Detection, Anomaly Detection, PII Detection, Polars, Schema Inference

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
4. **Statistical Rigor**: Implementation of well-established statistical methods for drift and anomaly detection
5. **Privacy Awareness**: Built-in PII detection and data masking capabilities
6. **Extensibility**: Modular architecture enabling seamless integration of custom validators

### 1.3 Contributions

This work presents:

- A unified data adapter layer supporting multiple input formats
- Optimized validation algorithms using Polars LazyFrame for memory-efficient processing
- Implementation of comprehensive statistical drift detection methods
- Advanced anomaly detection algorithms including ML-based approaches
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
│   (239 total)   │  │    (11 types)    │  │   (8 patterns)  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • Schema (14)   │  │ • KS Test       │  │ • Email         │
│ • Completeness  │  │ • Chi-Square    │  │ • Phone         │
│ • Uniqueness    │  │ • Wasserstein   │  │ • SSN           │
│ • Distribution  │  │ • PSI           │  │ • Credit Card   │
│ • String (17)   │  │ • Jensen-Shannon│  │ • Korean RRN    │
│ • Datetime (10) │  │ • Earth Mover   │  │ • Korean Phone  │
│ • Aggregate (8) │  │ • KL Divergence │  │ • Bank Account  │
│ • Multi-column  │  │ • Histogram     │  │ • Passport      │
│ • Anomaly (13)  │  │ • Cosine Sim    │  └────────┬────────┘
│ • Drift (11)    │  │ • Feature Drift │           │
│ • Geospatial    │  │ • Concept Drift │           │
│ • Query (5)     │  └────────┬────────┘           │
│ • Table (7)     │           │                    │
│ • Business (6)  │           │                    │
│ • Localization  │           │                    │
│ • ML Feature    │           │                    │
│ • Profiling (6) │           │                    │
│ • Referential   │           │                    │
│ • TimeSeries    │           │                    │
│ • Privacy (14)  │           │                    │
└────────┬────────┘           │                    │
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
- **Modular Extensibility**: Base classes and mixins enable rapid development of specialized validators

---

## 3. Core Components

### 3.1 Validators (239 Total)

Truthound provides **239 validators** across **21 categories**, offering comprehensive data quality coverage:

| Category | Count | Key Validators |
|----------|-------|----------------|
| **Schema** | 14 | `ColumnExistsValidator`, `ColumnTypeValidator`, `TableSchemaValidator`, `ReferentialIntegrityValidator` |
| **Completeness** | 7 | `NullValidator`, `NotNullValidator`, `CompletenessRatioValidator`, `ConditionalNullValidator` |
| **Uniqueness** | 13 | `UniqueValidator`, `DuplicateValidator`, `PrimaryKeyValidator`, `CompoundKeyValidator` |
| **Distribution** | 15 | `RangeValidator`, `BetweenValidator`, `OutlierValidator`, `KLDivergenceValidator`, `ChiSquareValidator` |
| **String** | 17 | `RegexValidator`, `EmailValidator`, `PhoneValidator`, `JsonSchemaValidator`, `LikePatternValidator` |
| **Datetime** | 10 | `DateFormatValidator`, `DateBetweenValidator`, `RecentDataValidator`, `DateutilParseableValidator` |
| **Aggregate** | 8 | `MeanBetweenValidator`, `MedianBetweenValidator`, `SumBetweenValidator`, `TypeValidator` |
| **Cross-table** | 4 | `CrossTableRowCountValidator`, `CrossTableAggregateValidator`, `CrossTableDistinctCountValidator` |
| **Multi-column** | 16 | `ColumnComparisonValidator`, `ConditionalValidator`, `MutualExclusivityValidator`, `CascadeNullValidator` |
| **Query** | 5 | `SQLExpressionValidator`, `CustomExpressionValidator`, `AggregateExpressionValidator` |
| **Table** | 7 | `RowCountValidator`, `ColumnCountValidator`, `DataFreshnessValidator`, `TableCompletenessValidator` |
| **Geospatial** | 6 | `LatitudeValidator`, `LongitudeValidator`, `CoordinatePairValidator`, `BoundingBoxValidator` |
| **Drift** | 11 | `KSTestValidator`, `ChiSquareDriftValidator`, `WassersteinValidator`, `PSIValidator`, `JensenShannonValidator` |
| **Anomaly** | 13 | `IsolationForestValidator`, `LOFValidator`, `MahalanobisValidator`, `IQRAnomalyValidator`, `DBSCANAnomalyValidator` |
| **Business Rule** | 6 | `LuhnValidator`, `ISBNValidator`, `CreditCardValidator`, `IBANValidator`, `VATValidator`, `SWIFTValidator` |
| **Localization** | 8 | `KoreanBusinessNumberValidator`, `KoreanRRNValidator`, `JapanesePostalCodeValidator`, `ChineseIDValidator` |
| **ML Feature** | 4 | `FeatureNullImpactValidator`, `FeatureScaleValidator`, `FeatureCorrelationMatrixValidator`, `TargetLeakageValidator` |
| **Profiling** | 6 | `CardinalityValidator`, `UniquenessRatioValidator`, `EntropyValidator`, `ValueFrequencyValidator` |
| **Referential** | 11 | `ForeignKeyValidator`, `CompositeForeignKeyValidator`, `OrphanRecordValidator`, `CircularReferenceValidator` |
| **Time Series** | 12 | `TimeSeriesGapValidator`, `TimeSeriesMonotonicValidator`, `SeasonalityValidator`, `TrendValidator` |
| **Privacy** | 14 | `GDPRComplianceValidator`, `CCPAComplianceValidator`, `GlobalPrivacyValidator`, `DataRetentionValidator` |

> **Detailed Documentation**: For comprehensive descriptions of each validator, including usage examples and configuration options, see **[Validator Reference (docs/VALIDATORS.md)](docs/VALIDATORS.md)**.

#### Key Features

- **`mostly` parameter**: All validators support partial pass rates (e.g., `mostly=0.95` allows 5% failures)
- **Statistical tests**: KL Divergence, Chi-Square, Kolmogorov-Smirnov for distribution validation
- **ML-based anomaly detection**: Isolation Forest, LOF, One-Class SVM, DBSCAN
- **SQL LIKE patterns**: `LikePatternValidator` supports `%` and `_` wildcards
- **Flexible date parsing**: `DateutilParseableValidator` handles multiple date formats automatically
- **Cross-table validation**: Compare row counts, aggregates between related tables
- **Geospatial validation**: Coordinate validation with bounding box support

### 3.2 Drift Detectors

| Detector | Method | Best For | Threshold |
|----------|--------|----------|-----------|
| `KSTestValidator` | Kolmogorov-Smirnov Test | Continuous numeric distributions | p-value < 0.05 |
| `ChiSquareDriftValidator` | Chi-Square Test | Categorical distributions | p-value < 0.05 |
| `WassersteinValidator` | Earth Mover's Distance | Distribution shape comparison | Context-dependent |
| `PSIValidator` | Population Stability Index | Model feature monitoring | PSI >= 0.1 (moderate), >= 0.25 (significant) |
| `JensenShannonValidator` | Jensen-Shannon Divergence | Any distribution (symmetric, bounded) | JS >= 0.1 |
| `KLDivergenceValidator` | Kullback-Leibler Divergence | Information loss measurement | KL > threshold |
| `HistogramDriftValidator` | Histogram Intersection | Visual distribution comparison | Intersection < 0.8 |
| `CosineSimilarityValidator` | Cosine Similarity | High-dimensional data | Similarity < 0.9 |
| `FeatureDriftValidator` | Multi-feature Analysis | Feature importance changes | Context-dependent |
| `ConceptDriftValidator` | Concept Change Detection | Label distribution shifts | Context-dependent |

### 3.3 Anomaly Detectors

| Detector | Method | Best For | Characteristics |
|----------|--------|----------|-----------------|
| `IsolationForestValidator` | Tree-based Isolation | High-dimensional data | No distribution assumptions |
| `LOFValidator` | Local Outlier Factor | Clustered data | Density-based detection |
| `OneClassSVMValidator` | Support Vector Machine | Complex boundaries | Kernel-based separation |
| `DBSCANAnomalyValidator` | Density Clustering | Noise detection | Cluster-based outliers |
| `MahalanobisValidator` | Covariance Distance | Multivariate normal data | Correlation-aware |
| `EllipticEnvelopeValidator` | Robust Gaussian | Contaminated data | Robust covariance estimation |
| `PCAAnomalyValidator` | Reconstruction Error | High-dimensional reduction | Principal component analysis |
| `IQRAnomalyValidator` | Interquartile Range | Univariate outliers | Distribution-free |
| `MADAnomalyValidator` | Median Absolute Deviation | Robust univariate | Resistant to extremes |
| `GrubbsTestValidator` | Statistical Test | Single outlier detection | Iterative removal |
| `TukeyFencesValidator` | Fence Classification | Inner/outer outliers | Traditional method |
| `PercentileAnomalyValidator` | Percentile Bounds | Custom thresholds | Flexible boundaries |
| `ZScoreMultivariateValidator` | Combined Z-scores | Multi-column analysis | Configurable aggregation |

### 3.4 Schema System

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

### 3.5 Auto Schema Caching

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
Lower Bound = Q1 - k * IQR
Upper Bound = Q3 + k * IQR
```

Where `k = 1.5` for standard outliers, `k = 3.0` for extreme outliers.

**Implementation Optimization**: Single-pass computation of Q1, Q3, and outlier counts for all numeric columns.

### 4.2 Kolmogorov-Smirnov Test

Measures maximum difference between empirical cumulative distribution functions:

```
D = max|F1(x) - F2(x)|
```

P-value approximation uses the asymptotic Kolmogorov distribution.

### 4.3 Population Stability Index (PSI)

Quantifies distribution shift between baseline and current populations:

```
PSI = sum((Pi - Qi) * ln(Pi / Qi))
```

Where Pi and Qi are proportions in bin i for baseline and current distributions.

**Industry Standard Interpretation**:
- PSI < 0.1: No significant change
- 0.1 <= PSI < 0.25: Moderate change
- PSI >= 0.25: Significant change

### 4.4 Chi-Square Test

Tests independence between observed and expected categorical frequencies:

```
chi2 = sum((Oi - Ei)^2 / Ei)
```

P-value computed using Wilson-Hilferty approximation.

### 4.5 Jensen-Shannon Divergence

Symmetric measure of distribution similarity (bounded [0, 1]):

```
JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
```

Where M = 0.5 * (P + Q) and KL is the Kullback-Leibler divergence.

### 4.6 Mahalanobis Distance

Multivariate distance accounting for correlations:

```
D = sqrt((x - mu)^T * Sigma^(-1) * (x - mu))
```

Where mu is the mean vector and Sigma is the covariance matrix.

### 4.7 Isolation Forest

Anomaly score based on path length in random trees:

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where h(x) is the path length and c(n) is the average path length for n samples.

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
| 5M vs 5M rows | 3.68s | 0.04s | **92x faster** |

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
| Unit Tests | 39 | Pass |
| Stress Tests | 53 | Pass |
| Extreme Stress Tests | 14 | Pass |
| Validator Tests (P0) | 32 | Pass |
| Validator Tests (P1) | 27 | Pass |
| Validator Tests (P2) | 27 | Pass |
| Drift Validator Tests | 52 | Pass |
| Anomaly Validator Tests | 31 | Pass |
| Multi-column Validator Tests | 43 | Pass |
| Query Validator Tests | 14 | Pass |
| Table Validator Tests | 21 | Pass |
| Geospatial Validator Tests | 26 | Pass |
| Business Rule Validator Tests | 22 | Pass |
| Localization Validator Tests | 28 | Pass |
| ML Feature Validator Tests | 23 | Pass |
| Profiling Validator Tests | 23 | Pass |
| Referential Validator Tests | 28 | Pass |
| Time Series Validator Tests | 30 | Pass |
| Privacy Validator Tests | 46 | Pass |
| Integration Tests | 138 | Pass |
| **Total** | **717** | **All Pass** |

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
| Korean RRN | `XXXXXX-XXXXXXX` | 98% |
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
| Zero Configuration | Yes | No | No | No |
| Polars Native | Yes | No | No | No |
| LazyFrame Support | Yes | No | No | No |
| Drift Detection | Yes (11 methods) | Plugin | No | Yes |
| Anomaly Detection | Yes (13 methods) | No | No | Limited |
| PII Detection | Yes | No | No | Yes |
| Schema Inference | Yes | Yes | Yes | Yes |
| Auto Caching | Yes | No | No | No |
| `mostly` Parameter | Yes | Yes | No | No |
| Cross-table Validation | Yes | Yes | No | Yes |
| Statistical Tests (KL, Chi2) | Yes | Yes | No | No |
| Geospatial Validation | Yes | No | No | No |
| Time Series Validation | Yes (12) | No | No | Limited |
| Referential Integrity | Yes (11) | Plugin | No | Yes |
| ML Feature Validation | Yes (4) | No | No | No |
| Privacy Compliance (GDPR/CCPA) | Yes (14) | No | No | Limited |
| Validator Count | 239 | 300+ | 50+ | 100+ |

### 8.2 Honest Assessment

**Strengths**:
1. Performance advantage from Polars (not unique to Truthound)
2. True zero-configuration with auto schema caching
3. Comprehensive drift detection with 11 statistical methods
4. Advanced anomaly detection including ML-based approaches
5. Korean-specific PII patterns
6. 239 validators covering most common data quality checks
7. Great Expectations-compatible `mostly` parameter
8. Geospatial coordinate validation
9. Time series validation with gap, seasonality, and trend detection
10. Referential integrity validation for complex data relationships
11. ML feature quality validation (leakage, correlation, scale)
12. Asian localization support (Korean, Japanese, Chinese)
13. Global privacy compliance (GDPR, CCPA, LGPD, PIPEDA, APPI)

**Limitations** (see Section 11):
1. No production deployment validation yet
2. No ecosystem integrations (Airflow, dbt, etc.)
3. Limited documentation and community

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
# Basic installation
pip install truthound

# With drift detection support (scipy)
pip install truthound[drift]

# With anomaly detection support (scipy + scikit-learn)
pip install truthound[anomaly]

# Full installation with all optional dependencies
pip install truthound[all]
```

### 9.3 Optional Dependencies

| Extra | Packages | Features |
|-------|----------|----------|
| `drift` | scipy | Statistical drift tests (KS, Chi-square, Wasserstein) |
| `anomaly` | scipy, scikit-learn | ML-based anomaly detection (Isolation Forest, LOF, SVM) |
| `all` | jinja2, pandas, scipy, scikit-learn | All optional features |
| `dev` | pytest, pytest-cov, ruff, mypy | Development tools |

### 9.4 Development Setup

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
    print("Warning: Significant drift detected!")

# Large dataset with sampling
drift = th.compare(
    "historical.parquet",
    "current.parquet",
    sample_size=10000  # 92x speedup
)

# Export for CI/CD
with open("drift_report.json", "w") as f:
    f.write(drift.to_json())
```

### 10.4 Anomaly Detection

```python
from truthound.validators.anomaly import (
    IsolationForestValidator,
    MahalanobisValidator,
    IQRAnomalyValidator,
)

# Isolation Forest for multi-dimensional anomalies
validator = IsolationForestValidator(
    columns=["feature1", "feature2", "feature3"],
    contamination=0.05,
    max_anomaly_ratio=0.1
)
issues = validator.validate(df.lazy())

# Mahalanobis distance for correlated features
validator = MahalanobisValidator(
    columns=["x", "y", "z"],
    threshold=3.0  # Chi-square threshold
)

# IQR-based detection for single columns
validator = IQRAnomalyValidator(
    column="value",
    k=1.5  # 1.5 for standard, 3.0 for extreme
)
```

### 10.5 PII Detection

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

1. **No Production Validation**: Untested in large-scale production environments
2. **Limited Integrations**: No native support for Airflow, dbt, Dagster, etc.
3. **Documentation**: Minimal API documentation and tutorials
4. **Community**: No established user community or support channels

### 11.2 Completed Improvements

- ~~**Phase 1**: Expand validator library (50+ validators)~~ **Completed** (239 validators)
- ~~**Phase 1.1**: Add drift detection validators~~ **Completed** (11 validators)
- ~~**Phase 1.2**: Add anomaly detection validators~~ **Completed** (13 validators)
- ~~**Phase 1.3**: Add multi-column validators~~ **Completed** (16 validators)
- ~~**Phase 1.4**: Add geospatial validators~~ **Completed** (6 validators)
- ~~**Phase 1.5**: Add business rule validators~~ **Completed** (6 validators)
- ~~**Phase 1.6**: Add localization validators~~ **Completed** (8 validators)
- ~~**Phase 1.7**: Add ML feature validators~~ **Completed** (4 validators)
- ~~**Phase 1.8**: Add profiling validators~~ **Completed** (6 validators)
- ~~**Phase 1.9**: Add referential integrity validators~~ **Completed** (11 validators)
- ~~**Phase 1.10**: Add time series validators~~ **Completed** (12 validators)
- ~~**Phase 1.11**: Add privacy compliance validators (GDPR/CCPA)~~ **Completed** (14 validators)

### 11.3 Planned Improvements

1. **Phase 2**: Add pipeline integrations (Airflow, Prefect)
2. **Phase 3**: Web dashboard for visualization
3. **Phase 4**: Database connectors (PostgreSQL, BigQuery)
4. **Phase 5**: Real-time streaming validation

---

## 12. References

1. Polars Documentation. https://pola.rs/
2. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
3. Pearson, K. (1900). "On the criterion that a given system of deviations..."
4. Lin, J. (1991). "Divergence measures based on the Shannon entropy"
5. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
6. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"
7. Mahalanobis, P. C. (1936). "On the generalized distance in statistics"
8. Great Expectations Documentation. https://greatexpectations.io/
9. Pandera Documentation. https://pandera.readthedocs.io/

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
- [scikit-learn](https://scikit-learn.org/) — Machine learning library (optional)
- [SciPy](https://scipy.org/) — Scientific computing library (optional)

---

<p align="center">
  <strong>Truthound — Your data's loyal guardian.</strong>
</p>
