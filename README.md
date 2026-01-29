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

> **Beta Release**: Core features are stable, APIs may still change in minor versions.

---

## Abstract

<img width="300" height="300" alt="Truthound_icon" src="https://github.com/user-attachments/assets/90d9e806-8895-45ec-97dc-f8300da4d997" />

Truthound is a data quality validation framework built on Polars, a Rust-based DataFrame library. The framework provides zero-configuration validation through automatic schema inference and supports a wide range of validation scenarios from basic schema checks to statistical drift detection.

[![PyPI version](https://img.shields.io/pypi/v/truthound.svg)](https://pypi.org/project/truthound/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Powered by Polars](https://img.shields.io/badge/Powered%20by-Polars-2563EB?logo=polars&logoColor=white)](https://pola.rs/)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/ddotta/awesome-polars)
[![Downloads](https://static.pepy.tech/badge/truthound?color=green)](https://pepy.tech/project/truthound)

**Documentation**: [Document Site](https://truthound.netlify.app/)

**Related Projects**

| Project | Description | Status |
|---------|-------------|--------|
| [truthound-orchestration](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration for Airflow, Dagster, Prefect, and dbt | Beta |
| [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) | Web-based data quality monitoring dashboard | Beta |

---

## Metrics

| Metric | Value |
|--------|-------|
| Test Cases | 8,259 |
| Validators | 264 |
| Validator Categories | 28 |

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
truthound compare baseline.csv current.csv  # Drift detection
truthound scan data.csv                     # PII scanning
truthound auto-profile data.csv             # Profiling
truthound new validator my_validator        # Code scaffolding
```

---

## CLI Reference

### Core Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `learn` | Learn schema from data | `--output`, `--no-constraints` |
| `check` | Validate data quality | `--validators`, `--min-severity`, `--schema`, `--strict`, `--format` |
| `scan` | Scan for PII | `--format`, `--output` |
| `mask` | Mask sensitive data | `--columns`, `--strategy` (redact/hash/fake), `--strict` |
| `profile` | Generate data profile | `--format`, `--output` |
| `compare` | Detect data drift | `--method` (auto/ks/psi/chi2/js), `--threshold`, `--strict` |

### Profiler Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `auto-profile` | Profile with auto-detection | `--patterns`, `--correlations`, `--sample`, `--top-n` |
| `generate-suite` | Generate validation rules from profile | `--strictness`, `--preset`, `--code-style` |
| `quick-suite` | Profile and generate rules in one step | `--strictness`, `--sample-size` |
| `list-formats` | List supported output formats | - |
| `list-presets` | List available presets | - |
| `list-categories` | List rule categories | - |

### Checkpoint Commands (CI/CD)

| Command | Description | Key Options |
|---------|-------------|-------------|
| `checkpoint run` | Run validation pipeline | `--config`, `--data`, `--strict`, `--slack`, `--webhook` |
| `checkpoint list` | List available checkpoints | `--config`, `--format` |
| `checkpoint validate` | Validate configuration | `--strict` |
| `checkpoint init` | Initialize sample config | `--output`, `--format` |

### ML Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `ml anomaly` | Detect anomalies | `--method` (zscore/iqr/mad/isolation_forest), `--contamination` |
| `ml drift` | Detect data drift | `--method` (distribution/feature/multivariate), `--threshold` |
| `ml learn-rules` | Learn validation rules | `--strictness`, `--min-confidence`, `--max-rules` |

### Docs Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `docs generate` | Generate HTML/PDF report | `--theme`, `--format` (html/pdf), `--title` |
| `docs themes` | List available themes | - |

### Lineage Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `lineage show` | Display lineage information | `--node`, `--direction` (upstream/downstream/both) |
| `lineage impact` | Analyze change impact | `--max-depth`, `--output` |
| `lineage visualize` | Generate lineage visualization | `--renderer` (d3/cytoscape/graphviz/mermaid), `--theme` |

### Realtime Commands (Streaming)

| Command | Description | Key Options |
|---------|-------------|-------------|
| `realtime validate` | Validate streaming data | `--batch-size`, `--max-batches` |
| `realtime monitor` | Monitor validation metrics | `--interval`, `--duration` |
| `realtime checkpoint list` | List checkpoints | `--dir` |
| `realtime checkpoint show` | Show checkpoint details | `--dir` |
| `realtime checkpoint delete` | Delete checkpoint | `--dir`, `--force` |

### Benchmark Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `benchmark run` | Run performance benchmarks | `--suite` (quick/ci/full), `--size`, `--iterations` |
| `benchmark list` | List available benchmarks | `--format` |
| `benchmark compare` | Compare benchmark results | `--threshold` |

### Scaffolding Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `new validator` | Create custom validator | `--template` (basic/column/pattern/range/comparison/composite/full) |
| `new reporter` | Create custom reporter | `--template` (basic/full), `--extension` |
| `new plugin` | Create plugin package | `--type` (validator/reporter/hook/datasource/action/full) |
| `new list` | List scaffold types | `--verbose` |
| `new templates` | List available templates | - |

### Plugin Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `plugin list` | List discovered plugins | `--type`, `--state`, `--verbose` |
| `plugin info` | Show plugin details | `--json` |
| `plugin load` | Load a plugin | `--activate/--no-activate` |
| `plugin unload` | Unload a plugin | - |
| `plugin enable` | Enable a plugin | - |
| `plugin disable` | Disable a plugin | - |
| `plugin create` | Create plugin template | `--type`, `--author` |

### Dashboard Command

| Command | Description | Key Options |
|---------|-------------|-------------|
| `dashboard` | Launch interactive dashboard | `--profile`, `--port`, `--host`, `--debug` |

---

## Python API Guides

### Validators

| Guide | Description |
|-------|-------------|
| [Categories](https://truthound.netlify.app/guides/validators/categories/) | 28 validator categories overview |
| [Built-in](https://truthound.netlify.app/guides/validators/built-in/) | 264 built-in validators reference |
| [Custom Validators](https://truthound.netlify.app/guides/validators/custom-validators/) | `@custom_validator` decorator, `ValidatorBuilder` fluent API |
| [Enterprise SDK](https://truthound.netlify.app/guides/validators/enterprise-sdk/) | Sandbox, signing, licensing, fuzzing |
| [Security](https://truthound.netlify.app/guides/validators/security/) | ReDoS protection, SQL injection prevention |
| [i18n](https://truthound.netlify.app/guides/validators/i18n/) | 7-language error messages |
| [Optimization](https://truthound.netlify.app/guides/validators/optimization/) | Expression batch execution, DAG parallel |

### Data Sources

| Guide | Description |
|-------|-------------|
| [Files](https://truthound.netlify.app/guides/datasources/files/) | CSV, JSON, Parquet, NDJSON, JSONL |
| [Databases](https://truthound.netlify.app/guides/datasources/databases/) | PostgreSQL, MySQL, SQLite, Oracle, SQL Server |
| [Cloud Warehouses](https://truthound.netlify.app/guides/datasources/cloud-warehouses/) | BigQuery, Snowflake, Redshift, Databricks |
| [Streaming](https://truthound.netlify.app/guides/datasources/streaming/) | Kafka, Kinesis, Pub/Sub adapters |
| [Custom Sources](https://truthound.netlify.app/guides/datasources/custom-sources/) | IDataSource protocol implementation |

### Profiler

| Guide | Description |
|-------|-------------|
| [Basics](https://truthound.netlify.app/guides/profiler/basics/) | Column statistics, distribution analysis |
| [Patterns](https://truthound.netlify.app/guides/profiler/patterns/) | Email, phone, credit card detection |
| [Rule Generation](https://truthound.netlify.app/guides/profiler/rule-generation/) | Auto-generate validation rules |
| [Drift Detection](https://truthound.netlify.app/guides/profiler/drift-detection/) | KS, PSI, Chi-square, Wasserstein |
| [Quality Scoring](https://truthound.netlify.app/guides/profiler/quality-scoring/) | Data quality metrics |
| [Sampling](https://truthound.netlify.app/guides/profiler/sampling/) | Block, multi-stage, progressive sampling |
| [Caching](https://truthound.netlify.app/guides/profiler/caching/) | xxhash fingerprint-based caching |
| [ML Inference](https://truthound.netlify.app/guides/profiler/ml-inference/) | ML-based rule generation |
| [Threshold Tuning](https://truthound.netlify.app/guides/profiler/threshold-tuning/) | Automatic threshold optimization |
| [Visualization](https://truthound.netlify.app/guides/profiler/visualization/) | Profile visualization |
| [i18n](https://truthound.netlify.app/guides/profiler/i18n/) | Localized profiling output |
| [Schema Evolution](https://truthound.netlify.app/guides/profiler/schema-evolution/) | Change detection, compatibility analysis |
| [Distributed](https://truthound.netlify.app/guides/profiler/distributed/) | Spark, Dask, Ray backends |
| [Enterprise Sampling](https://truthound.netlify.app/guides/profiler/enterprise-sampling/) | 100M+ row sampling strategies |

### Data Docs

| Guide | Description |
|-------|-------------|
| [HTML Reports](https://truthound.netlify.app/guides/datadocs/html-reports/) | Report generation pipeline |
| [Charts](https://truthound.netlify.app/guides/datadocs/charts/) | ApexCharts, Chart.js, Plotly.js, SVG |
| [Sections](https://truthound.netlify.app/guides/datadocs/sections/) | Report sections configuration |
| [Themes](https://truthound.netlify.app/guides/datadocs/themes/) | 6 built-in themes, white-labeling |
| [Versioning](https://truthound.netlify.app/guides/datadocs/versioning/) | 4 versioning strategies |
| [PDF Export](https://truthound.netlify.app/guides/datadocs/pdf-export/) | Chunked rendering, parallel processing |
| [Custom Renderers](https://truthound.netlify.app/guides/datadocs/custom-renderers/) | Jinja2, String, File, Callable templates |
| [Dashboard](https://truthound.netlify.app/guides/datadocs/dashboard/) | Interactive dashboard integration |

### Reporters

| Guide | Description |
|-------|-------------|
| [Console](https://truthound.netlify.app/guides/reporters/console/) | Rich terminal output |
| [JSON/YAML](https://truthound.netlify.app/guides/reporters/json-yaml/) | Structured output formats |
| [HTML/Markdown](https://truthound.netlify.app/guides/reporters/html-markdown/) | Document formats |
| [CI Reporters](https://truthound.netlify.app/guides/reporters/ci-reporters/) | JUnit, GitHub Actions, GitLab CI |
| [Custom SDK](https://truthound.netlify.app/guides/reporters/custom-sdk/) | IReporter protocol implementation |

### Storage

| Guide | Description |
|-------|-------------|
| [Filesystem](https://truthound.netlify.app/guides/stores/filesystem/) | Local file storage |
| [Cloud Storage](https://truthound.netlify.app/guides/stores/cloud-storage/) | S3, GCS, Azure Blob |
| [Versioning](https://truthound.netlify.app/guides/stores/versioning/) | Incremental, Semantic, Timestamp, GitLike |
| [Retention](https://truthound.netlify.app/guides/stores/retention/) | Time, Count, Size, Status, Tag policies |
| [Tiering](https://truthound.netlify.app/guides/stores/tiering/) | Hot/Warm/Cold/Archive migration |
| [Caching](https://truthound.netlify.app/guides/stores/caching/) | LRU, LFU, TTL backends |
| [Replication](https://truthound.netlify.app/guides/stores/replication/) | Sync/Async/Semi-Sync cross-region |
| [Observability](https://truthound.netlify.app/guides/stores/observability/) | Audit, Metrics, Tracing |

### Checkpoint & CI/CD

| Guide | Description |
|-------|-------------|
| [Basics](https://truthound.netlify.app/guides/checkpoint/basics/) | Checkpoint configuration |
| [Triggers](https://truthound.netlify.app/guides/checkpoint/triggers/) | Event-based triggering |
| [Actions](https://truthound.netlify.app/guides/checkpoint/actions/) | Notifications, Webhook, Storage, Incident |
| [Routing](https://truthound.netlify.app/guides/checkpoint/routing/) | Python + Jinja2 rule engine, 11 built-in rules |
| [Deduplication](https://truthound.netlify.app/guides/checkpoint/deduplication/) | InMemory/Redis, 4 window strategies |
| [Throttling](https://truthound.netlify.app/guides/checkpoint/throttling/) | Token Bucket, Fixed/Sliding Window |
| [Escalation](https://truthound.netlify.app/guides/checkpoint/escalation/) | Multi-level policies, state machine |
| [Async](https://truthound.netlify.app/guides/checkpoint/async/) | Celery, Ray, Kubernetes backends |
| [CI Platforms](https://truthound.netlify.app/guides/checkpoint/ci-platforms/) | GitHub Actions, GitLab CI, Jenkins, etc. |

### Configuration

| Guide | Description |
|-------|-------------|
| [Environment Variables](https://truthound.netlify.app/guides/configuration/environment-vars/) | Environment-based configuration |
| [Sources](https://truthound.netlify.app/guides/configuration/sources/) | Data source configuration |
| [Datasource Config](https://truthound.netlify.app/guides/configuration/datasource-config/) | Connection settings |
| [Store Config](https://truthound.netlify.app/guides/configuration/store-config/) | Storage backend settings |
| [Profiler Config](https://truthound.netlify.app/guides/configuration/profiler-config/) | Profiler settings |
| [Checkpoint Config](https://truthound.netlify.app/guides/configuration/checkpoint-config/) | CI/CD pipeline settings |
| [Logging](https://truthound.netlify.app/guides/configuration/logging/) | JSON format, ELK/Loki integration |
| [Metrics](https://truthound.netlify.app/guides/configuration/metrics/) | Prometheus counters, gauges, histograms |
| [Audit](https://truthound.netlify.app/guides/configuration/audit/) | Operation trail, compliance reporting |
| [Encryption](https://truthound.netlify.app/guides/configuration/encryption/) | AES-256-GCM, Cloud KMS integration |
| [Resilience](https://truthound.netlify.app/guides/configuration/resilience/) | Circuit breaker, retry, bulkhead |

### Advanced

| Guide | Description |
|-------|-------------|
| [ML Anomaly](https://truthound.netlify.app/guides/advanced/ml-anomaly/) | Isolation Forest, LOF, One-Class SVM |
| [Lineage](https://truthound.netlify.app/guides/advanced/lineage/) | DAG tracking, OpenLineage integration |
| [Plugins](https://truthound.netlify.app/guides/advanced/plugins/) | Security sandbox, signing, hot reload |
| [Performance](https://truthound.netlify.app/guides/advanced/performance/) | Optimization strategies |

---

## Validator Categories

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
| security | SQL injection, ReDoS protection |
| sdk | Custom validator development |
| timeout | Distributed timeout management |
| i18n | Internationalized error messages |
| streaming | Streaming data validation |
| memory | Memory-aware processing |
| optimization | Execution optimization |

---

## Data Sources

| Category | Sources |
|----------|---------|
| DataFrame | Polars, Pandas, PySpark |
| Core SQL | PostgreSQL, MySQL, SQLite |
| Cloud DW | BigQuery, Snowflake, Redshift, Databricks |
| Enterprise | Oracle, SQL Server |
| File | CSV, Parquet, JSON, NDJSON |
| Streaming | Kafka, Kinesis, Pub/Sub |

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
