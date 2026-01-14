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

| Feature | Description |
|---------|-------------|
| **Zero Configuration** | Automatic schema inference with fingerprint-based caching (xxhash) |
| **289 Validators** | 28 categories including schema, completeness, uniqueness, distribution, drift, anomaly |
| **Polars LazyFrame** | Native Polars operations, expression-based batch execution, single collect() optimization |
| **DAG Parallel Execution** | Dependency-aware orchestration with 3 execution strategies (Sequential, Parallel, Adaptive) |
| **Custom Validator SDK** | `@custom_validator` decorator, `ValidatorBuilder` fluent API, testing utilities, 7 templates |
| **i18n Error Messages** | 7 languages (EN, KO, JA, ZH, DE, FR, ES) with message catalogs |
| **Privacy Compliance** | GDPR, CCPA, LGPD, PIPEDA, APPI support with PII detection and masking |
| **ReDoS Protection** | Regex safety analysis, ML-based prediction (sklearn), CVE database, RE2 engine support |
| **Distributed Timeout** | Deadline propagation, cascading timeout, graceful degradation |
| **Enterprise Sampling** | Block, multi-stage, column-aware, progressive sampling for 100M+ rows |
| **Performance Profiling** | Validator-level timing, memory, throughput metrics with Prometheus export |

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

### Auto-Profiling

| Feature | Description |
|---------|-------------|
| **Statistical Profiling** | Column-level statistics, distribution analysis, missing value detection |
| **Pattern Detection** | Email, phone, credit card, custom regex patterns |
| **Rule Generation** | Auto-generate validation rules from profile with `Suite.execute()` |
| **Distributed Processing** | Spark, Dask, Ray, Local backends |
| **Incremental Scheduling** | Cron, interval, data change triggers |
| **Schema Evolution** | Change detection, compatibility analysis (FULL, BACKWARD, FORWARD, NONE) |
| **Unified Resilience** | Circuit breaker, retry, bulkhead, rate limiter with fluent builder |

### Data Docs (HTML Reports)

| Feature | Description |
|---------|-------------|
| **6 Built-in Themes** | Default, Light, Dark, Minimal, Modern, Professional |
| **4 Chart Libraries** | ApexCharts, Chart.js, Plotly.js, SVG (zero-dependency) |
| **15 Languages** | EN, KO, JA, ZH, DE, FR, ES, PT, IT, RU, AR, TH, VI, ID, TR |
| **White-labeling** | Enterprise themes with custom branding, logo, colors |
| **PDF Export** | Chunked rendering, parallel processing for large reports |
| **Report Versioning** | 4 strategies with diff and rollback support |

### Plugin Architecture

| Feature | Description |
|---------|-------------|
| **Security Sandbox** | NoOp, Process, Container isolation engines |
| **Plugin Signing** | HMAC, RSA, Ed25519 algorithms with trust store |
| **6 Security Presets** | DEVELOPMENT, TESTING, STANDARD, ENTERPRISE, STRICT, AIRGAPPED |
| **Version Constraints** | Semver-based (^, ~, >=, <, ranges) |
| **Dependency Management** | Graph-based resolution, cycle detection, topological sort |
| **Hot Reload** | File watching, graceful reload with rollback |
| **Documentation** | AST-based extraction, Markdown/HTML/JSON renderers |
| **CLI Extension** | Entry point based plugin system (`truthound.cli` group) |

### Checkpoint & CI/CD

| Feature | Description |
|---------|-------------|
| **Saga Pattern** | 8 compensation strategies (Backward, Forward, Semantic, Pivot, etc.) |
| **12 CI Platforms** | GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure DevOps, etc. |
| **9 Notification Providers** | Slack, Email, PagerDuty, GitHub, Webhook, Teams, OpsGenie, Discord, Telegram |
| **GitHub OIDC** | 30+ claims parsing, AWS/GCP/Azure/Vault credential exchange |
| **4 Distributed Backends** | Local, Celery, Ray, Kubernetes |
| **Circuit Breaker** | 3 states (CLOSED, OPEN, HALF_OPEN), failure detection strategies |
| **Idempotency** | Request fingerprinting, duplicate detection, TTL expiration |

### Advanced Notifications

| Feature | Description |
|---------|-------------|
| **Rule-based Routing** | Python expression + Jinja2 engine, 11 built-in rules, combinators (AllOf, AnyOf, NotRule) |
| **Deduplication** | InMemory/Redis Streams backends, 4 window strategies (Sliding, Tumbling, Session, Adaptive) |
| **Rate Limiting** | Token Bucket, Fixed/Sliding Window, 5 throttler types with builder pattern |
| **Escalation Policies** | Multi-level escalation, state machine, 3 storage backends (InMemory, Redis, SQLite) |

### ML & Lineage

| Feature | Description |
|---------|-------------|
| **6 Anomaly Algorithms** | Isolation Forest, LOF, One-Class SVM, DBSCAN, Statistical, Autoencoders |
| **4 Drift Algorithms** | KS Test, Chi-Square, PSI, Jensen-Shannon Divergence |
| **ML Model Monitoring** | Performance metrics, quality metrics, drift detection, alerting |
| **Lineage Graph** | DAG-based dependency tracking, column-level lineage, impact analysis |
| **4 Visualization Renderers** | D3.js, Cytoscape.js, Graphviz, Mermaid |
| **OpenLineage Integration** | Industry-standard lineage events, run lifecycle management |

### Storage Features

| Feature | Description |
|---------|-------------|
| **Cloud Storage** | S3, GCS, Azure Blob backends with connection pooling |
| **Versioning** | 4 strategies (Incremental, Semantic, Timestamp, GitLike) |
| **Retention** | 6 policies (Time, Count, Size, Status, Tag, Composite) |
| **Tiering** | Hot/Warm/Cold/Archive with 5 migration policies |
| **Caching** | LRU, LFU, TTL backends with 4 cache modes |
| **Replication** | Sync/Async/Semi-Sync cross-region with conflict resolution |
| **Backpressure** | 6 strategies with circuit breaker and monitoring |
| **Batch Optimization** | Memory-aware buffer, async batch writer |

### Enterprise Infrastructure

| Component | Features |
|-----------|----------|
| **Logging** | JSON format, correlation IDs, ELK/Loki/Fluentd integration, async logging |
| **Metrics** | Prometheus counters, gauges, histograms, HTTP endpoint, push gateway |
| **Config** | Environment profiles (dev/staging/prod), Vault/AWS Secrets integration, hot reload |
| **Audit** | Full operation trail, Elasticsearch/S3/Kafka storage, compliance reporting (SOC2/GDPR/HIPAA) |
| **Encryption** | AES-256-GCM, ChaCha20-Poly1305, field-level encryption, Cloud KMS (AWS/GCP/Azure/Vault) |

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
