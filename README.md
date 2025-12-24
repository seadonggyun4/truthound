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
8. [Storage & Reporting](#8-storage--reporting)
   - [8.3 Data Sources & Execution Engines](#83-data-sources--execution-engines)
9. [Comparative Analysis](#9-comparative-analysis)
10. [Installation](#10-installation)
11. [Usage Examples](#11-usage-examples)
12. [Limitations and Future Work](#12-limitations-and-future-work)
13. [References](#13-references)

**Additional Documentation:**
- [Data Sources Usage (docs/DATASOURCES.md)](docs/DATASOURCES.md)
- [Data Sources Architecture (docs/DATASOURCES_ARCHITECTURE.md)](docs/DATASOURCES_ARCHITECTURE.md)
- [Validators Reference (docs/VALIDATORS.md)](docs/VALIDATORS.md)
- [Statistical Methods (docs/STATISTICAL_METHODS.md)](docs/STATISTICAL_METHODS.md)
- [Storage Backends (docs/STORES.md)](docs/STORES.md)
- [Reporters (docs/REPORTERS.md)](docs/REPORTERS.md)
- [Usage Examples (docs/EXAMPLES.md)](docs/EXAMPLES.md)
- [Test Coverage (docs/TEST_COVERAGE.md)](docs/TEST_COVERAGE.md)

---

## 1. Introduction

### 1.1 Problem Statement

Data quality issues represent a significant challenge in modern data engineering workflows. According to industry reports, data scientists spend approximately 60-80% of their time on data preparation and cleaning tasks. Traditional data quality tools often require extensive configuration, suffer from performance limitations when processing large datasets, and lack native support for modern columnar data formats.

### 1.2 Design Goals

Truthound was designed with the following objectives:

1. **Zero Configuration**: Immediate usability without boilerplate setup code
2. **High Performance**: Leveraging Rust-based Polars for computational efficiency
3. **Universal Input Support**: Native handling of diverse data formats
4. **Multi-Backend Support**: Unified abstraction for Polars, Pandas, SQL, and Spark
5. **Statistical Rigor**: Implementation of well-established statistical methods for drift and anomaly detection
6. **Privacy Awareness**: Built-in PII detection and data masking capabilities
7. **Extensibility**: Modular architecture enabling seamless integration of custom validators

### 1.3 Contributions

This work presents:

- A unified data source abstraction supporting Polars, Pandas, SQL databases, and Spark
- Enterprise data sources: BigQuery, Snowflake, Redshift, Databricks, Oracle, SQL Server
- Execution engines with backend-specific optimizations (SQL pushdown, lazy evaluation)
- Cost-aware query execution for cloud data warehouses (BigQuery dry-run cost estimation)
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

Truthound implements well-established statistical methods for drift detection, anomaly detection, and distribution analysis.

> **Detailed Documentation**: For comprehensive explanations of each method including formulas, interpretation guidelines, and implementation examples, see **[Statistical Methods (docs/STATISTICAL_METHODS.md)](docs/STATISTICAL_METHODS.md)**.

| Method | Use Case | Key Metric |
|--------|----------|------------|
| **IQR** | Univariate outliers | Q1 - 1.5×IQR to Q3 + 1.5×IQR |
| **Kolmogorov-Smirnov** | Distribution comparison | D statistic, p-value |
| **PSI** | Model monitoring | PSI < 0.1 (stable), ≥ 0.25 (significant) |
| **Chi-Square** | Categorical drift | χ² statistic, p-value |
| **Jensen-Shannon** | Symmetric divergence | JS ∈ [0, 1] |
| **Mahalanobis** | Multivariate outliers | Distance threshold |
| **Isolation Forest** | ML-based anomaly | Anomaly score |

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

Truthound maintains comprehensive test coverage with **1004 tests** across all validation features.

> **Detailed Documentation**: For complete test suite information, including stress tests, extreme stress tests, and PII detection coverage, see **[Test Coverage (docs/TEST_COVERAGE.md)](docs/TEST_COVERAGE.md)**.

| Category | Tests | Status |
|----------|-------|--------|
| Core Tests (Unit, Stress, Extreme) | 106 | All Pass |
| Validator Tests (P0-P2, All Categories) | 473 | All Pass |
| Integration Tests | 138 | All Pass |
| Storage & Reporter Tests | 98 | All Pass |
| Data Source Tests (SQL, Enterprise) | 194 | All Pass |
| **Total** | **1004** | **All Pass** |

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

## 8. Storage & Reporting

Truthound provides enterprise-ready infrastructure for persisting validation results and generating reports.

### 8.1 Storage Backends

Store validation results across different backends for tracking, auditing, and trend analysis.

```python
from truthound.stores import get_store, ValidationResult
import truthound as th

# Create store
store = get_store("filesystem", base_path=".truthound/results")
store.initialize()

# Save validation results
report = th.check("data.csv")
result = ValidationResult.from_report(report, "data.csv")
run_id = store.save(result)

# Retrieve and query results
retrieved = store.get(run_id)
all_runs = store.list_ids()
```

| Backend | Package | Description |
|---------|---------|-------------|
| `filesystem` | (built-in) | Local JSON storage with optional compression |
| `memory` | (built-in) | In-memory storage for testing |
| `s3` | boto3 | AWS S3 storage |
| `gcs` | google-cloud-storage | Google Cloud Storage |
| `database` | sqlalchemy | SQL database (PostgreSQL, MySQL, SQLite) |

> **Detailed Documentation**: See **[Storage Backends (docs/STORES.md)](docs/STORES.md)** for configuration options, cloud setup, and custom backend implementation.

### 8.2 Report Formats

Generate validation reports in multiple formats.

```python
from truthound.reporters import get_reporter

# JSON for API integration
json_reporter = get_reporter("json")
json_reporter.write(result, "report.json")

# HTML for web dashboards
html_reporter = get_reporter("html", title="Quality Report")
html_reporter.write(result, "report.html")

# Console for terminal output
console_reporter = get_reporter("console", color=True)
console_reporter.report(result)

# Markdown for documentation
md_reporter = get_reporter("markdown")
md_reporter.write(result, "REPORT.md")
```

| Format | Package | Use Case |
|--------|---------|----------|
| `json` | (built-in) | API integration, programmatic access |
| `console` | rich | Terminal output, debugging |
| `markdown` | (built-in) | Documentation, GitHub/GitLab |
| `html` | jinja2 | Web dashboards, email reports |

> **Detailed Documentation**: See **[Reporters (docs/REPORTERS.md)](docs/REPORTERS.md)** for customization, templates, and integration examples.

### 8.3 Data Sources & Execution Engines

Truthound supports 10+ data backends through a unified abstraction layer.

| Category | Sources | Features |
|----------|---------|----------|
| **DataFrame** | Polars, Pandas, PySpark | Native operations, auto-sampling |
| **Core SQL** | PostgreSQL, MySQL, SQLite | Connection pooling, SQL pushdown |
| **Cloud DW** | BigQuery, Snowflake, Redshift, Databricks | Cost control, IAM auth |
| **Enterprise** | Oracle, SQL Server | Windows auth, TNS support |
| **File** | CSV, Parquet, JSON, NDJSON | Lazy loading, streaming |

> **Usage Guide**: See **[Data Sources (docs/DATASOURCES.md)](docs/DATASOURCES.md)** for examples and configuration.
>
> **Architecture Deep Dive**: See **[Data Sources Architecture (docs/DATASOURCES_ARCHITECTURE.md)](docs/DATASOURCES_ARCHITECTURE.md)** for design patterns, extensibility guide, and quality assessment.

### 8.4 Architecture Overview

> **Detailed Documentation**: For comprehensive architecture documentation including design patterns, type system, and extension points, see **[Architecture (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)**.

---

## 9. Comparative Analysis

### 9.1 Feature Comparison

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

### 9.2 Honest Assessment

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

## 10. Installation

### 10.1 Requirements

- Python 3.11+
- Polars 1.x
- PyYAML
- Rich (for console output)
- Typer (for CLI)

### 10.2 Installation

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

### 10.3 Optional Dependencies

| Extra | Packages | Features |
|-------|----------|----------|
| `drift` | scipy | Statistical drift tests (KS, Chi-square, Wasserstein) |
| `anomaly` | scipy, scikit-learn | ML-based anomaly detection (Isolation Forest, LOF, SVM) |
| `all` | jinja2, pandas, scipy, scikit-learn | All optional features |
| `dev` | pytest, pytest-cov, ruff, mypy | Development tools |
| `s3` | boto3 | AWS S3 storage backend |
| `gcs` | google-cloud-storage | Google Cloud Storage backend |
| `database` | sqlalchemy | SQL database storage backend |
| `bigquery` | google-cloud-bigquery | Google BigQuery data source |
| `snowflake` | snowflake-connector-python | Snowflake data source |
| `redshift` | redshift-connector | Amazon Redshift data source |
| `databricks` | databricks-sql-connector | Databricks SQL data source |
| `oracle` | oracledb | Oracle Database data source |
| `sqlserver` | pyodbc / pymssql | SQL Server data source |
| `enterprise` | (all enterprise sources) | All enterprise data backends |

### 10.4 Development Setup

```bash
git clone https://github.com/seadonggyun4/Truthound.git
cd Truthound
pip install hatch
hatch env create
hatch run test
```

---

## 11. Usage Examples

> **Detailed Documentation**: For comprehensive examples including cross-table validation, time series validation, privacy compliance, CI/CD integration, and custom validators, see **[Usage Examples (docs/EXAMPLES.md)](docs/EXAMPLES.md)**.

### Quick Start

```python
import truthound as th

# Basic validation
report = th.check("data.csv")

# Schema-based validation
schema = th.learn("baseline.csv")
report = th.check("new_data.csv", schema=schema)

# Drift detection
drift = th.compare("train.csv", "production.csv")

# PII scanning and masking
pii_report = th.scan(df)
masked_df = th.mask(df, strategy="hash")
```

### CLI Quick Start

```bash
truthound check data.csv                    # Validate
truthound check data.csv --strict           # CI/CD mode
truthound compare baseline.csv current.csv  # Drift detection
truthound scan data.csv                     # PII scanning
```

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **No Production Validation**: Untested in large-scale production environments
2. **Limited Integrations**: No native support for Airflow, dbt, Dagster, etc.
3. **Documentation**: Minimal API documentation and tutorials
4. **Community**: No established user community or support channels

### 12.2 Completed Improvements

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
- ~~**Phase 4**: Storage backends & reporters infrastructure~~ **Completed** (5 backends, 4 formats)
- ~~**Phase 5**: Enterprise data sources (BigQuery, Snowflake, etc.)~~ **Completed** (6 sources)

### 12.3 Planned Improvements

1. **Phase 6**: Add pipeline integrations (Airflow, Prefect)
2. **Phase 7**: Web dashboard for visualization
3. **Phase 8**: Real-time streaming validation

---

## 13. References

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
