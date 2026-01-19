# Guides

Comprehensive guides for using Truthound effectively in production environments.

## Core Guides

### [Validators](validators.md)

Complete reference for all 289 validators across 28 categories. Learn about:

- Completeness validators (null checks, missing data)
- Uniqueness validators (duplicates, primary keys)
- Format validators (email, phone, URL patterns)
- Range validators (numeric bounds, date ranges)
- Custom validator development

### [Data Sources](datasources.md)

Connect Truthound to various data backends:

- File formats (CSV, Parquet, JSON, NDJSON)
- SQL databases (PostgreSQL, MySQL, SQLite)
- Cloud warehouses (BigQuery, Snowflake, Redshift, Databricks)
- Enterprise databases (Oracle, SQL Server)
- Apache Spark integration

### [CI/CD Integration](ci-cd.md)

Integrate data quality checks into your pipelines:

- Checkpoint configuration
- GitHub Actions, GitLab CI, Jenkins
- Slack, Email, PagerDuty notifications
- Saga pattern for distributed validation
- Historical analytics and monitoring

### [Configuration](configuration.md)

Comprehensive configuration reference:

- Environment configuration (dev/staging/prod)
- Configuration sources (files, env vars, Vault, AWS Secrets)
- DataSource and Store configuration
- Resilience patterns (circuit breaker, retry, bulkhead)
- Observability settings (audit, metrics, tracing)

### [Performance](performance.md)

Optimize Truthound for large-scale data:

- Benchmarks and performance targets
- Memory optimization techniques
- Parallel execution strategies
- Query pushdown for SQL sources
- Streaming mode for large files

## Feature Guides

### [Data Profiling](profiler.md)

Automatic data profiling and rule generation:

- Statistical profiling
- Pattern detection
- Schema inference
- Validation suite generation
- Incremental profiling

### [Data Docs](datadocs.md)

Generate beautiful HTML reports:

- 5 built-in themes
- 4 chart libraries
- PDF export
- Custom templates
- White-labeling

### [Reporters](reporters.md)

Output validation results in various formats:

- Console reporter
- JSON reporter
- HTML reporter
- JUnit reporter (CI/CD)
- Custom reporter SDK

### [Storage Backends](stores.md)

Persist validation results:

- Filesystem storage
- S3, GCS, Azure Blob
- Database storage
- Versioning and retention
- Cross-region replication

## Quick Links

| Topic | Guide |
|-------|-------|
| All validators | [Validators Reference](validators.md) |
| Database connections | [Data Sources](datasources.md) |
| Pipeline integration | [CI/CD Integration](ci-cd.md) |
| Settings & options | [Configuration](configuration.md) |
| Large datasets | [Performance](performance.md) |
| Auto-profiling | [Profiler](profiler.md) |
| HTML reports | [Data Docs](datadocs.md) |
