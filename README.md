<div align="center">
  <img width="500" height="500" alt="logo" src="https://github.com/user-attachments/assets/4b4dea6c-46b9-49e5-af19-744a3b216bf8" />
</div>

<h1 align="center">Truthound</h1>

<p align="center">
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
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
<img width="300" height="300" alt="Truthound_icon" src="https://github.com/user-attachments/assets/90d9e806-8895-45ec-97dc-f8300da4d997" />

Truthound is a high-performance data quality validation framework designed for modern data engineering pipelines. The library leverages the computational efficiency of Polars—a Rust-based DataFrame library—to achieve order-of-magnitude performance improvements over traditional Python-based validation solutions.

**Keywords**: Data Quality, Data Validation, Statistical Drift Detection, Anomaly Detection, PII Detection, Polars, Schema Inference

---

## Key Features

| Feature | Description |
|---------|-------------|
| **265+ Validators** | Schema, completeness, uniqueness, distribution, string patterns, datetime, and more |
| **Zero Configuration** | Automatic schema inference with fingerprint-based caching |
| **High Performance** | Polars LazyFrame architecture for memory-efficient processing |
| **Statistical Analysis** | 11 drift detection methods, 15 anomaly detection algorithms |
| **Privacy Compliance** | GDPR, CCPA, LGPD, PIPEDA, APPI pattern detection |
| **Multi-Backend Support** | Polars, Pandas, SQL databases, Spark, and cloud data warehouses |
| **CI/CD Integration** | Native support for 12 CI platforms with checkpoint orchestration |
| **Auto-Profiling** | Automatic rule generation from data profiling |
| **Data Docs** | Interactive HTML reports with 5 themes and 4 chart libraries |
| **Plugin Architecture** | Extensible system for custom validators, reporters, and datasources |
| **ML Integration** | Anomaly detection, drift detection, and rule learning |
| **Data Lineage** | Graph-based lineage tracking and impact analysis |
| **Realtime Validation** | Streaming support with Kafka, Kinesis, and Pub/Sub |

---

## Quick Start

### Installation

```bash
# Basic installation
pip install truthound

# With all optional features
pip install truthound[all]
```

### Python API

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

# Statistical profiling
profile = th.profile("data.csv")
```

### CLI

```bash
truthound check data.csv                    # Validate
truthound check data.csv --strict           # CI/CD mode
truthound compare baseline.csv current.csv  # Drift detection
truthound scan data.csv                     # PII scanning
truthound auto-profile data.csv -o profile.json  # Profiling
truthound docs generate profile.json -o report.html  # HTML report
```

---

## Performance

| Operation | 10M Rows | Throughput |
|-----------|----------|------------|
| `th.check()` | 3.53s | 2.83M rows/sec |
| `th.profile()` | 0.15s | 66.7M rows/sec |
| `th.learn()` | 0.27s | 37.0M rows/sec |

Drift detection with sampling achieves **92x speedup** on 5M row datasets.

---

## Documentation

### Getting Started
- **[Getting Started Guide](docs/GETTING_STARTED.md)** — Installation, quick start, and basic usage

### Core Concepts
- **[Architecture Overview](docs/ARCHITECTURE.md)** — System design and core principles
- **[Validators Reference](docs/VALIDATORS.md)** — Complete reference for all 265+ validators
- **[Statistical Methods](docs/STATISTICAL_METHODS.md)** — Mathematical foundations for drift and anomaly detection

### Features by Phase

| Phase | Documentation | Description |
|-------|---------------|-------------|
| **Phase 1-3** | [Core Validators](docs/VALIDATORS.md) | 265 validators across 21 categories |
| **Phase 4** | [Storage & Reporters](docs/STORES.md), [Reporters](docs/REPORTERS.md) | Persistence and output formats |
| **Phase 5** | [Data Sources](docs/DATASOURCES.md) | Multi-backend support (BigQuery, Snowflake, etc.) |
| **Phase 6** | [Checkpoint & CI/CD](docs/CHECKPOINT.md) | Orchestration and CI/CD integration |
| **Phase 7** | [Auto-Profiling](docs/PROFILER.md) | Automatic rule generation |
| **Phase 8** | [Data Docs](docs/DATADOCS.md) | HTML report generation |
| **Phase 9** | [Plugin Architecture](docs/PLUGINS.md) | Extensibility framework |
| **Phase 10** | [Advanced Features](docs/ADVANCED.md) | ML, Lineage, and Realtime modules |

### Reference
- **[API Reference](docs/API_REFERENCE.md)** — Complete API documentation
- **[Examples](docs/EXAMPLES.md)** — Usage examples and patterns
- **[Test Coverage](docs/TEST_COVERAGE.md)** — 1004 tests across all features

---

## Validator Categories

| Category | Count | Description |
|----------|-------|-------------|
| Schema | 14 | Column structure, types, relationships |
| Completeness | 7 | Null detection, required fields |
| Uniqueness | 13 | Duplicates, primary keys, composite keys |
| Distribution | 15 | Range, outliers, statistical tests |
| String | 18 | Regex, email, URL, JSON validation |
| Datetime | 10 | Format, range, sequence validation |
| Aggregate | 8 | Mean, median, sum constraints |
| Cross-table | 4 | Multi-table relationships |
| Multi-column | 21 | Column comparisons, conditional logic |
| Query | 20 | SQL/Polars expression validation |
| Table | 18 | Row count, freshness, metadata |
| Geospatial | 9 | Coordinates, bounding boxes |
| Drift | 13 | KS, PSI, Chi-square, Wasserstein |
| Anomaly | 15 | IQR, Z-score, Isolation Forest, LOF |
| Business | 8 | Luhn, IBAN, VAT, ISBN validation |
| Localization | 9 | Korean, Japanese, Chinese identifiers |
| ML Feature | 5 | Leakage detection, correlation |
| Profiling | 7 | Cardinality, entropy, frequency |
| Referential | 13 | Foreign keys, orphan records |
| Time Series | 14 | Gaps, seasonality, trend detection |
| Privacy | 15 | GDPR, CCPA, LGPD compliance |

---

## Data Sources

| Category | Sources |
|----------|---------|
| **DataFrame** | Polars, Pandas, PySpark |
| **Core SQL** | PostgreSQL, MySQL, SQLite |
| **Cloud DW** | BigQuery, Snowflake, Redshift, Databricks |
| **Enterprise** | Oracle, SQL Server |
| **File** | CSV, Parquet, JSON, NDJSON |

---

## Installation Options

```bash
# Core installation
pip install truthound

# Feature-specific extras
pip install truthound[drift]      # Drift detection (scipy)
pip install truthound[anomaly]    # Anomaly detection (scikit-learn)
pip install truthound[pdf]        # PDF export (weasyprint)
pip install truthound[dashboard]  # Interactive dashboard (reflex)

# Data source extras
pip install truthound[bigquery]   # Google BigQuery
pip install truthound[snowflake]  # Snowflake
pip install truthound[redshift]   # Amazon Redshift
pip install truthound[databricks] # Databricks
pip install truthound[oracle]     # Oracle Database
pip install truthound[sqlserver]  # SQL Server
pip install truthound[enterprise] # All enterprise sources

# Full installation
pip install truthound[all]
```

---

## Comparative Analysis

| Feature | Truthound | Great Expectations | Pandera |
|---------|-----------|-------------------|---------|
| Zero Configuration | Yes | No | No |
| Polars Native | Yes | No | No |
| LazyFrame Support | Yes | No | No |
| Drift Detection | 13 methods | Plugin | No |
| Anomaly Detection | 15 methods | No | No |
| PII Detection | Yes | No | No |
| Cross-table Validation | Yes | Yes | No |
| Geospatial Validation | Yes | No | No |
| Time Series Validation | 14 validators | No | No |
| Privacy Compliance | GDPR/CCPA/LGPD | No | No |
| Validator Count | 265+ | 300+ | 50+ |

---

## Requirements

- Python 3.11+
- Polars 1.x
- PyYAML
- Rich (console output)
- Typer (CLI)

---

## Development

```bash
git clone https://github.com/seadonggyun4/Truthound.git
cd Truthound
pip install hatch
hatch env create
hatch run test
```

---

## References

1. Polars Documentation. https://pola.rs/
2. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
4. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"

---

## License

MIT License — Copyright (c) 2024-2025 Truthound Contributors

---

## Acknowledgments

Built with [Polars](https://pola.rs/), [Rich](https://rich.readthedocs.io/), [Typer](https://typer.tiangolo.com/), [scikit-learn](https://scikit-learn.org/), and [SciPy](https://scipy.org/).

---

<p align="center">
  <strong>Truthound — Your data's loyal guardian.</strong>
</p>
