<div align="center">
  <img width="500px" alt="Truthound Banner" src="https://raw.githubusercontent.com/seadonggyun4/Truthound/main/docs/assets/truthound_banner.png" />
</div>

<h1 align="center">Truthound</h1>

<p align="center">
  <strong>Zero-Configuration Data Quality Framework Powered by Polars</strong>
</p>

<p align="center">
  <em>Sniffs out bad data</em>
</p>

---

## Abstract

<img width="300" height="300" alt="Truthound_icon" src="https://github.com/user-attachments/assets/90d9e806-8895-45ec-97dc-f8300da4d997" />

Truthound is a data quality validation framework built on Polars, a Rust-based DataFrame library. The framework provides zero-configuration validation through automatic schema inference and supports a wide range of validation scenarios from basic schema checks to statistical drift detection.

[![PyPI version](https://img.shields.io/pypi/v/truthound.svg)](https://pypi.org/project/truthound/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://img.shields.io/pepy/dt/truthound?color=brightgreen)](https://pepy.tech/project/truthound)

**Documentation**: [https://truthound.netlify.app](https://truthound.netlify.app/)

**Related Projects**
| Project | Description |
|---------|-------------|
| [truthound-orchestration](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration for Airflow, Dagster, Prefect, and dbt |
| [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) | Web-based data quality monitoring dashboard |

---

## Implementation Status

### Verified Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Test Cases | 7,613 | Collected via pytest |
| Validators | 289 | Validator classes available |
| Validator Categories | 28 | Distinct subdirectories |

### Core Features

| Feature | Status | Description |
|---------|--------|-------------|
| Zero Configuration | Implemented | Automatic schema inference with fingerprint-based caching |
| Polars LazyFrame | Implemented | Native Polars operations for all core validations |
| DAG Parallel Execution | Implemented | Dependency-aware validator orchestration |
| Custom Validator SDK | Implemented | Decorators, fluent builder, testing utilities |
| Privacy Compliance | Implemented | GDPR, CCPA, LGPD, PIPEDA, APPI support |
| ReDoS Protection | Implemented | Regex safety analysis, ML-based prediction, safe execution |
| Plugin Architecture | Implemented | Security sandbox, code signing, version constraints, hot reload |
| Multi-Backend Support | Implemented | Polars, Pandas, SQL databases, cloud data warehouses |

---

## Quick Start

### Installation

```bash
pip install truthound

# With optional features
pip install truthound[all]
```

### Python API

```python
import truthound as th

# Basic validation
report = th.check("data.csv")

# Parallel validation (DAG-based execution)
report = th.check("data.csv", parallel=True, max_workers=4)

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

# Code scaffolding
truthound new validator my_validator        # Create validator
truthound new reporter json_export          # Create reporter
truthound new plugin my_plugin              # Create plugin
```

---

## Validator Categories

The following validator categories are implemented:

| Category | Description |
|----------|-------------|
| schema | Column structure, types, relationships |
| completeness | Null detection, required fields |
| uniqueness | Duplicates, primary keys, composite keys |
| distribution | Range, outliers, statistical tests |
| string | Regex, email, URL, JSON validation |
| datetime | Format, range, sequence validation |
| aggregate | Mean, median, sum constraints |
| cross_table | Multi-table relationships |
| multi_column | Column comparisons, conditional logic |
| query | SQL/Polars expression validation |
| table | Row count, freshness, metadata |
| geospatial | Coordinates, bounding boxes |
| drift | KS, PSI, Chi-square, Wasserstein |
| anomaly | IQR, Z-score, Isolation Forest, LOF |
| business_rule | Luhn, IBAN, VAT, ISBN validation |
| localization | Korean, Japanese, Chinese identifiers |
| ml_feature | Leakage detection, correlation |
| profiling | Cardinality, entropy, frequency |
| referential | Foreign keys, orphan records |
| timeseries | Gaps, seasonality, trend detection |
| privacy | PII detection and compliance rules |
| security | SQL injection prevention, ReDoS protection |
| sdk | Custom validator development tools |
| timeout | Distributed timeout management |
| i18n | Internationalized error messages |
| streaming | Streaming data validation |
| memory | Memory-aware processing |
| optimization | Validator execution optimization |

---

## Data Sources

| Category | Sources |
|----------|---------|
| DataFrame | Polars, Pandas, PySpark |
| Core SQL | PostgreSQL, MySQL, SQLite |
| Cloud DW | BigQuery, Snowflake, Redshift, Databricks |
| Enterprise | Oracle, SQL Server |
| File | CSV, Parquet, JSON, NDJSON |

---

## Streaming Support

### Protocol-based Adapters

Kafka and Kinesis adapters implement `IStreamSource`/`IStreamSink` protocols with async operations:

| Adapter | Library | Features |
|---------|---------|----------|
| KafkaAdapter | aiokafka | Consumer groups, partition management, SASL/SSL authentication |
| KinesisAdapter | aiobotocore | Multi-shard consumption, enhanced fan-out, checkpointing |

### StreamingSource Pattern

File-based and cloud streaming sources use `StreamingSource` base class:

| Source | Description |
|--------|-------------|
| ParquetSource | Parquet file streaming |
| CSVSource | CSV file streaming |
| JSONLSource | JSON Lines streaming |
| ArrowIPCSource | Arrow IPC format |
| ArrowFlightSource | Arrow Flight protocol |
| PubSubSource | Google Cloud Pub/Sub (requires google-cloud-pubsub) |

Note: Kafka and Kinesis use protocol-based adapters with `aiokafka` and `aiobotocore`. Pub/Sub uses the `StreamingSource` pattern with synchronous operations.

---

## Enterprise Features

### Custom Validator SDK

```python
from truthound.validators.sdk import custom_validator, ValidatorBuilder

@custom_validator(name="check_positive", category="numeric")
def check_positive(df, column: str, strict: bool = True):
    values = df[column]
    return (values > 0).all() if strict else (values >= 0).all()

validator = (
    ValidatorBuilder("revenue_check")
    .with_category("business")
    .with_column("revenue")
    .with_condition(lambda df: df["revenue"] > 0)
    .build()
)
```

### ReDoS Protection

```python
from truthound.validators.security.redos import RegexComplexityAnalyzer, SafeRegexExecutor

analyzer = RegexComplexityAnalyzer()
result = analyzer.analyze(r"(a+)+$")
print(f"Safe: {result.is_safe}, Score: {result.complexity_score}")

executor = SafeRegexExecutor(timeout=1.0)
match = executor.match(pattern, text)
```

### Plugin Architecture

```python
from truthound.plugins import create_enterprise_manager, SecurityPolicyPresets

manager = create_enterprise_manager(
    security_preset=SecurityPolicyPresets.ENTERPRISE,
)
await manager.load("my-plugin", verify_signature=True)
result = await manager.execute_in_sandbox("my-plugin", my_function, arg1, arg2)
```

### CLI Extension System

External packages can extend the truthound CLI via entry points:

```toml
# pyproject.toml
[project.entry-points."truthound.cli"]
serve = "my_package.cli:register_commands"
```

```python
# my_package/cli.py
import typer

def register_commands(app: typer.Typer) -> None:
    @app.command(name="serve")
    def serve(port: int = 8765) -> None:
        """Start my service."""
        typer.echo(f"Starting on port {port}")
```

After installation, the command is available:

```bash
pip install my-package
truthound serve --port 9000
```

### Enterprise Infrastructure

| Component | Features |
|-----------|----------|
| Logging | JSON format, correlation IDs, ELK/Loki/Fluentd integration |
| Metrics | Prometheus counters, gauges, histograms |
| Config | Environment profiles, Vault/AWS Secrets integration |
| Audit | Operation tracking, Elasticsearch/S3/Kafka storage |
| Encryption | AES-256-GCM, Cloud KMS integration |

### Storage Features

| Feature | Description |
|---------|-------------|
| Cloud Storage | S3, GCS, Azure Blob backends |
| Versioning | 4 strategies (Incremental, Semantic, Timestamp, GitLike) |
| Retention | TTL policies, automatic cleanup |
| Tiering | Hot/Warm/Cold/Archive with automatic migration |
| Caching | LRU, LFU, TTL with multiple cache modes |
| Replication | Sync/Async/Semi-Sync cross-region replication |
| Backpressure | Memory, queue depth, latency-based strategies |

---

## Documentation

| Phase | Description |
|-------|-------------|
| Phase 1-3 | Core Validators |
| Phase 4 | Storage and Reporters |
| Phase 5 | Multi-DataSource Support |
| Phase 6 | Checkpoint and CI/CD |
| Phase 7 | Auto-Profiling |
| Phase 8 | Data Docs |
| Phase 9 | Plugin Architecture |
| Phase 10 | ML, Lineage, Realtime |

### Phase 11-14: Enterprise Features

| Phase | Name | Status | Repository |
|-------|------|--------|------------|
| Phase 11 | Workflow Integration | Complete | [truthound-orchestration](https://github.com/seadonggyun4/truthound-orchestration) |
| Phase 12 | Web UI and REST API | Complete | [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) |
| Phase 13 | Business Glossary and Data Catalog | Complete | [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) |
| Phase 14 | Advanced Notifications | Planned | truthound |

---

## Installation Options

```bash
# Core installation
pip install truthound

# Feature-specific extras
pip install truthound[drift]      # Drift detection (scipy)
pip install truthound[anomaly]    # Anomaly detection (scikit-learn)
pip install truthound[pdf]        # PDF export (weasyprint)

# Data source extras
pip install truthound[bigquery]   # Google BigQuery
pip install truthound[snowflake]  # Snowflake
pip install truthound[redshift]   # Amazon Redshift
pip install truthound[databricks] # Databricks
pip install truthound[oracle]     # Oracle Database
pip install truthound[sqlserver]  # SQL Server

# Security extras
pip install truthound[encryption] # Encryption (cryptography)

# Full installation
pip install truthound[all]
```

---

## Requirements

- Python 3.11+
- Polars 1.x
- PyYAML
- Rich
- Typer

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

Apache License 2.0

---

## Acknowledgments

Built with Polars, Rich, Typer, scikit-learn, and SciPy.
