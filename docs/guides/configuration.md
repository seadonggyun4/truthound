# Configuration Guide

This document provides comprehensive configuration reference for Truthound, covering all configurable components from environment setup to enterprise integrations.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Configuration](#environment-configuration)
3. [Configuration Sources](#configuration-sources)
4. [DataSource Configuration](#datasource-configuration)
5. [Store Configuration](#store-configuration)
6. [Checkpoint Configuration](#checkpoint-configuration)
7. [Profiler Configuration](#profiler-configuration)
8. [Resilience Configuration](#resilience-configuration)
9. [Observability Configuration](#observability-configuration)
10. [Environment Variables Reference](#environment-variables-reference)
11. [Configuration File Formats](#configuration-file-formats)
12. [Best Practices](#best-practices)

---

## Overview

Truthound uses a layered configuration system that supports multiple sources with priority-based merging:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Configuration Priority                            │
├─────────────────────────────────────────────────────────────────────┤
│  Priority 200: Secret Managers (Vault, AWS Secrets)                 │
│  Priority 100: Environment Variables (TRUTHOUND_*)                  │
│  Priority  30: Local Config (local.yaml)                           │
│  Priority  20: Environment Config (production.yaml, staging.yaml)   │
│  Priority  10: Base Config (base.yaml)                             │
└─────────────────────────────────────────────────────────────────────┘
```

Higher priority sources override lower priority values.

---

## Environment Configuration

### Environment Detection

Truthound automatically detects the runtime environment:

```python
from truthound.infrastructure.config import Environment

# Auto-detection from environment variables
env = Environment.current()

# Available environments
Environment.DEVELOPMENT  # Local development
Environment.TESTING      # Test environments
Environment.STAGING      # Pre-production
Environment.PRODUCTION   # Production environment
Environment.LOCAL        # Local overrides
```

### Environment Detection Order

1. `TRUTHOUND_ENV` environment variable
2. `ENVIRONMENT` environment variable
3. `ENV` environment variable
4. Default: `DEVELOPMENT`

### Environment Properties

```python
env = Environment.current()

env.is_production   # True if PRODUCTION or STAGING
env.is_development  # True if DEVELOPMENT, LOCAL, or TESTING
```

---

## Configuration Sources

### FileConfigSource

Load configuration from YAML, JSON, or TOML files:

```python
from truthound.infrastructure.config import ConfigManager, FileConfigSource

manager = ConfigManager()

# Add file source (priority 10-30)
manager.add_source(FileConfigSource(
    path="config/base.yaml",
    required=True,
    watch=True,  # Enable file watching for hot reload
))
```

**Supported formats:**
- YAML (`.yaml`, `.yml`)
- JSON (`.json`)
- TOML (`.toml`)

### EnvConfigSource

Load configuration from environment variables:

```python
from truthound.infrastructure.config import EnvConfigSource

# Priority 100 by default
manager.add_source(EnvConfigSource(
    prefix="TRUTHOUND",
    separator="_",
))

# Environment variables mapping:
# TRUTHOUND_DATABASE_HOST -> database.host
# TRUTHOUND_LOGGING_LEVEL -> logging.level
```

### VaultConfigSource

Load secrets from HashiCorp Vault:

```python
from truthound.infrastructure.config import VaultConfigSource

manager.add_source(VaultConfigSource(
    url="https://vault.example.com",
    token="${VAULT_TOKEN}",  # Or from environment
    path="secret/data/truthound",
    namespace="default",
))
```

### AwsSecretsSource

Load secrets from AWS Secrets Manager:

```python
from truthound.infrastructure.config import AwsSecretsSource

manager.add_source(AwsSecretsSource(
    secret_name="truthound/config",
    region_name="us-east-1",  # Or AWS_REGION env var
))
```

---

## DataSource Configuration

### DataSourceConfig

Base configuration for all data sources:

```python
from truthound.datasources import DataSourceConfig

config = DataSourceConfig(
    name="my_dataset",            # Optional identifier
    max_rows=10_000_000,          # Maximum rows (default: 10M)
    max_memory_mb=4096,           # Memory limit in MB (default: 4GB)
    sample_size=100_000,          # Default sample size
    sample_seed=42,               # Reproducible sampling
    cache_schema=True,            # Cache schema information
    strict_types=False,           # Strict type checking
    metadata={},                  # Custom metadata
)
```

### SQL DataSource Configuration

```python
from truthound.datasources.sql import SQLDataSourceConfig

config = SQLDataSourceConfig(
    pool_size=5,                  # Connection pool size
    pool_timeout=30.0,            # Timeout for acquiring connection
    query_timeout=300.0,          # Query execution timeout
    fetch_size=10000,             # Rows to fetch at a time
)
```

### Creating DataSources

```python
from truthound.datasources import get_datasource, get_sql_datasource

# Auto-detect from data
source = get_datasource("data.csv")
source = get_datasource(polars_df)
source = get_datasource(pandas_df)

# SQL databases
source = get_sql_datasource(
    "postgresql://user:pass@localhost/db",
    table="users",
)
```

---

## Store Configuration

### StoreConfig

Configuration for validation result storage:

```python
from truthound.stores import StoreConfig

config = StoreConfig(
    namespace="default",           # Isolation scope
    prefix="",                     # Path prefix
    serialization_format="json",   # json, yaml, pickle
    compression=None,              # gzip, zstd, lz4
    metadata={},                   # Custom metadata
)
```

### Store Factory

```python
from truthound.stores import get_store, list_available_backends

# Available backends
backends = list_available_backends()
# ['filesystem', 'memory', 's3', 'gcs', 'azure', 'database']

# Create store
store = get_store("filesystem", base_path="./results")
store = get_store("s3", bucket="my-bucket", prefix="truthound/")
store = get_store("gcs", bucket="my-bucket", project="my-project")
```

### Backend-Specific Options

#### Filesystem Store

```python
store = get_store(
    "filesystem",
    base_path="./truthound_results",
    create_dirs=True,
)
```

#### S3 Store

```python
store = get_store(
    "s3",
    bucket="my-bucket",
    prefix="truthound/",
    region="us-east-1",
    # Uses default AWS credential chain
)
```

#### GCS Store

```python
store = get_store(
    "gcs",
    bucket="my-bucket",
    project="my-gcp-project",
    prefix="truthound/",
    # Uses Application Default Credentials
)
```

#### Azure Blob Store

```python
store = get_store(
    "azure",
    container="my-container",
    connection_string="${AZURE_STORAGE_CONNECTION_STRING}",
    # Or account_url + credential
)
```

---

## Checkpoint Configuration

### CheckpointConfig

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig

config = CheckpointConfig(
    name="production_validation",
    data_source="s3://bucket/data.parquet",
    validators=["null", "duplicate", "range"],
    min_severity="medium",
    schema="schema.yaml",
    auto_schema=False,
    run_name_template="%Y%m%d_%H%M%S",
    tags={"env": "production", "team": "data-platform"},
    metadata={"owner": "data-team@company.com"},
    fail_on_critical=True,
    fail_on_high=False,
    timeout_seconds=3600,
    sample_size=100000,
)

checkpoint = Checkpoint(config=config)
```

### YAML Configuration

```yaml
# truthound.yaml
checkpoints:
- name: daily_data_validation
  data_source: data/production.csv
  validators:
  - 'null'
  - duplicate
  - range
  - regex
  validator_config:
    regex:
      patterns:
        email: ^[\w.+-]+@[\w-]+\.[\w.-]+$
        product_code: ^[A-Z]{2,4}[-_][0-9]{3,6}$
        phone: ^(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$
    range:
      columns:
        age:
          min_value: 0
          max_value: 150
        price:
          min_value: 0
  min_severity: medium
  auto_schema: true
  tags:
    environment: production
    team: data-platform
  actions:
  - type: store_result
    store_path: ./truthound_results
    partition_by: date
  - type: update_docs
    site_path: ./truthound_docs
    include_history: true
  - type: slack
    webhook_url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
    notify_on: failure
    channel: '#data-quality'
  triggers:
  - type: schedule
    interval_hours: 24
    run_on_weekdays: [0, 1, 2, 3, 4]  # Mon-Fri
```

### Route Configuration

Rule-based notification routing:

```yaml
# routing.yaml
routes:
  - name: critical_alerts
    rule:
      type: all_of
      rules:
        - type: severity
          min_severity: critical
        - type: tag
          tags:
            env: prod
    actions:
      - type: pagerduty
        routing_key: "${PAGERDUTY_KEY}"
    priority: critical
    enabled: true

  - name: slack_warnings
    rule:
      type: severity
      min_severity: high
    actions:
      - type: slack
        webhook_url: "${SLACK_WEBHOOK}"
        channel: "#data-quality"
    priority: high
```

**Available Rule Types:**

| Type | Description |
|------|-------------|
| `always` | Always matches |
| `never` | Never matches |
| `severity` | Match by severity level |
| `issue_count` | Match by issue count threshold |
| `status` | Match by checkpoint status |
| `tag` | Match by tags |
| `data_asset` | Match by data asset name |
| `metadata` | Match by metadata fields |
| `time_window` | Match within time window |
| `pass_rate` | Match by pass rate threshold |
| `error` | Match on errors |
| `expression` | Python expression |
| `template` | Jinja2 template |
| `all_of` | All rules must match (AND) |
| `any_of` | Any rule must match (OR) |
| `not` | Negate a rule |

---

## Profiler Configuration

### SuiteGeneratorConfig

Configuration for validation suite generation:

```python
from truthound.profiler import SuiteGeneratorConfig

config = SuiteGeneratorConfig(
    # Strictness
    strictness="medium",          # strict, medium, loose

    # Categories
    categories=CategoryConfig(
        include=["schema", "completeness", "uniqueness"],
        exclude=["anomaly"],
        priority_order=["schema", "completeness"],
    ),

    # Confidence
    confidence=ConfidenceConfig(
        min_level="medium",       # low, medium, high
        include_rationale=True,
        show_in_output=True,
    ),

    # Output
    output=OutputConfig(
        format="yaml",            # yaml, json, python, toml, checkpoint
        include_metadata=True,
        include_summary=True,
        group_by_category=True,
        sort_rules=True,
    ),
)
```

### Configuration Presets

```python
from truthound.profiler import SuiteGeneratorConfig, ConfigPreset

# Using string preset names
config = SuiteGeneratorConfig.from_preset("default")
config = SuiteGeneratorConfig.from_preset("strict")
config = SuiteGeneratorConfig.from_preset("loose")
config = SuiteGeneratorConfig.from_preset("minimal")
config = SuiteGeneratorConfig.from_preset("comprehensive")
config = SuiteGeneratorConfig.from_preset("schema_only")
config = SuiteGeneratorConfig.from_preset("format_only")
config = SuiteGeneratorConfig.from_preset("ci_cd")
config = SuiteGeneratorConfig.from_preset("development")
config = SuiteGeneratorConfig.from_preset("production")

# Using ConfigPreset enum (recommended for type safety)
config = SuiteGeneratorConfig.from_preset(ConfigPreset.DEFAULT)
config = SuiteGeneratorConfig.from_preset(ConfigPreset.STRICT)
config = SuiteGeneratorConfig.from_preset(ConfigPreset.CI_CD)
```

| Preset | Strictness | Min Confidence | Output Format | Use Case |
|--------|------------|----------------|---------------|----------|
| `default` | medium | low | yaml | General use |
| `strict` | strict | medium | yaml | Production validation |
| `loose` | loose | low | yaml | Exploratory analysis |
| `minimal` | loose | high | yaml | Quick schema check (fast mode) |
| `comprehensive` | strict | low | yaml | Deep analysis (full mode) |
| `schema_only` | medium | medium | yaml | Schema and completeness only |
| `format_only` | medium | medium | yaml | Format and pattern only |
| `ci_cd` | medium | medium | checkpoint | Pipeline integration |
| `development` | loose | low | python | Local development |
| `production` | strict | high | yaml | Production deployment |

### Dashboard Configuration

```python
from truthound.profiler.dashboard import DashboardConfig

config = DashboardConfig(
    host="localhost",
    port=8501,
    theme="light",                # light, dark, auto
    title="Truthound Data Profiler",
    enable_uploads=True,
    max_upload_size=200,          # MB
    cache_ttl=3600,               # seconds
    enable_export=True,
    show_raw_data=True,
    default_sample_size=100000,
    wide_layout=True,
    sidebar_state="expanded",
    enable_comparison=True,
    enable_recommendations=True,
    enable_ml_inference=True,
    require_auth=False,
    allowed_extensions=[".csv", ".parquet", ".json", ".xlsx", ".feather"],
)
```

---

## Resilience Configuration

### CircuitBreakerConfig

```python
from truthound.common.resilience import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,           # Failures to open circuit
    success_threshold=3,           # Successes to close circuit
    timeout_seconds=30.0,          # Time before half-open
    half_open_max_calls=3,         # Test calls in half-open
    failure_rate_threshold=50.0,   # Failure rate % to open
    slow_call_threshold_ms=1000.0, # Slow call definition
    slow_call_rate_threshold=50.0, # Slow call % to open
    window_size=100,               # Measurement window
    record_slow_calls=True,
)

# Presets
config = CircuitBreakerConfig.aggressive()      # Quick failure
config = CircuitBreakerConfig.lenient()         # Tolerant
config = CircuitBreakerConfig.for_database()    # DB optimized
config = CircuitBreakerConfig.for_external_api() # API optimized
```

### RetryConfig

```python
from truthound.common.resilience import RetryConfig

config = RetryConfig(
    max_attempts=3,
    base_delay=0.1,                # seconds
    max_delay=30.0,                # seconds
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.5,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(ValueError, TypeError, KeyError),
)

# Presets
config = RetryConfig.no_retry()
config = RetryConfig.quick()          # Fast retries
config = RetryConfig.persistent()     # Many retries
config = RetryConfig.exponential()    # Exponential backoff
```

### BulkheadConfig

```python
from truthound.common.resilience import BulkheadConfig

config = BulkheadConfig(
    max_concurrent=10,             # Max concurrent calls
    max_wait_time=0.0,             # Wait time for slot
    fairness=True,                 # FIFO ordering
)

# Presets
config = BulkheadConfig.small()        # 5 concurrent
config = BulkheadConfig.medium()       # 10 concurrent
config = BulkheadConfig.large()        # 25 concurrent
config = BulkheadConfig.for_database() # DB optimized
```

### RateLimiterConfig

```python
from truthound.common.resilience import RateLimiterConfig

config = RateLimiterConfig(
    rate=100,                      # Requests per period
    period_seconds=1.0,            # Period duration
    burst_size=None,               # Burst allowance
    algorithm="token_bucket",      # token_bucket, sliding_window, fixed_window
)

# Presets
config = RateLimiterConfig.per_second(rate=100, burst=150)
config = RateLimiterConfig.per_minute(rate=1000)
config = RateLimiterConfig.per_hour(rate=10000)
```

### ResilienceBuilder

Combine multiple resilience patterns:

```python
from truthound.common.resilience import ResilienceBuilder

wrapper = (
    ResilienceBuilder("my-service")
    .with_circuit_breaker(CircuitBreakerConfig.for_external_api())
    .with_retry(RetryConfig.exponential())
    .with_bulkhead(BulkheadConfig.medium())
    .with_rate_limit(RateLimiterConfig.per_second(100))
    .build()
)

# Execute with all resilience patterns
result = wrapper.execute(my_function, arg1, arg2)

# Or use as decorator
@wrapper
def risky_operation():
    return external_service.call()
```

**Builder Methods:**

| Method | Description |
|--------|-------------|
| `ResilienceBuilder(name)` | Create builder with service name |
| `.with_circuit_breaker(config)` | Add circuit breaker |
| `.with_retry(config)` | Add retry policy |
| `.with_bulkhead(config)` | Add bulkhead |
| `.with_rate_limit(config)` | Add rate limiter |
| `.build()` | Build the `ResilientWrapper` |

**Convenience Factory Methods:**

```python
# Pre-configured for database operations
wrapper = ResilienceBuilder.for_database("db-ops")

# Pre-configured for external API calls
wrapper = ResilienceBuilder.for_external_api("api-client")

# Simple defaults (circuit breaker + retry)
wrapper = ResilienceBuilder.simple("my-service")
```

---

## Observability Configuration

### AuditConfig

```python
from truthound.stores.observability import AuditConfig, AuditLogLevel

config = AuditConfig(
    enabled=True,
    level=AuditLogLevel.STANDARD,  # minimal, standard, verbose, debug
    backend="json",                # memory, file, json, elasticsearch, kafka
    file_path="./audit.log",
    include_data_preview=False,
    max_data_preview_size=1024,
    redact_sensitive=True,
    sensitive_fields=["password", "secret", "token", "api_key", "ssn", "credit_card"],
    retention_days=90,
    batch_size=100,
    flush_interval_seconds=5.0,
)
```

### MetricsConfig

```python
from truthound.stores.observability import MetricsConfig, MetricsExportFormat

config = MetricsConfig(
    enabled=True,
    export_format=MetricsExportFormat.PROMETHEUS,
    prefix="truthound_store",
    labels={"service": "data-quality"},
    enable_http_server=True,
    http_port=9090,
    http_path="/metrics",
    push_gateway_url=None,         # For push-based metrics
    push_interval_seconds=10.0,
    include_timestamps=True,
)
```

### TracingConfig

```python
from truthound.stores.observability import TracingConfig, TracingSampler

config = TracingConfig(
    enabled=True,
    service_name="truthound-store",
    sampler=TracingSampler.RATIO,
    sample_ratio=0.1,              # 10% sampling
    exporter="otlp",               # otlp, jaeger, zipkin, console
    endpoint="http://localhost:4317",
    headers={},
    propagators=["tracecontext", "baggage"],
    record_exceptions=True,
    max_attributes=128,
    max_events=128,
)
```

### ObservabilityConfig

Combined observability configuration:

```python
from truthound.stores.observability import ObservabilityConfig

# Full configuration
config = ObservabilityConfig(
    audit=AuditConfig(...),
    metrics=MetricsConfig(...),
    tracing=TracingConfig(...),
    correlation_id_header="X-Correlation-ID",
    environment="production",
    version="1.0.0",
)

# Presets
config = ObservabilityConfig.disabled()     # All disabled
config = ObservabilityConfig.minimal()      # Dev-friendly
config = ObservabilityConfig.production()   # Full observability
```

---

## Environment Variables Reference

### Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_ENV` | Environment (development, staging, production) | development |
| `TRUTHOUND_LOGGING_LEVEL` | Log level (TRACE, DEBUG, INFO, WARNING, ERROR) | INFO |
| `TRUTHOUND_LOGGING_FORMAT` | Log format (console, json, logfmt) | console |
| `TRUTHOUND_VALIDATION_TIMEOUT` | Validation timeout in seconds | 300 |
| `TRUTHOUND_VALIDATION_MAX_WORKERS` | Maximum worker threads | 4 |

### Database Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_DATABASE_HOST` | Database host | localhost |
| `TRUTHOUND_DATABASE_PORT` | Database port | 5432 |
| `TRUTHOUND_DATABASE_POOL_SIZE` | Connection pool size | 10 |

### Metrics Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_METRICS_ENABLED` | Enable metrics | true |
| `TRUTHOUND_METRICS_PORT` | Metrics HTTP port | 9090 |

### Suite Generator

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_SUITE_STRICTNESS` | Suite strictness level | medium |
| `TRUTHOUND_SUITE_MIN_CONFIDENCE` | Minimum confidence level | medium |
| `TRUTHOUND_SUITE_FORMAT` | Output format | yaml |
| `TRUTHOUND_SUITE_INCLUDE_CATEGORIES` | Categories to include (comma-separated) | all |
| `TRUTHOUND_SUITE_EXCLUDE_CATEGORIES` | Categories to exclude (comma-separated) | none |

### Secret Managers

| Variable | Description |
|----------|-------------|
| `VAULT_TOKEN` | HashiCorp Vault token |
| `AWS_REGION` | AWS region for Secrets Manager |

---

## Configuration File Formats

### YAML Configuration

```yaml
# config/base.yaml
logging:
  level: INFO
  format: console

metrics:
  enabled: true
  port: 9090

database:
  host: localhost
  port: 5432
  pool_size: 10

validation:
  timeout: 300
  max_workers: 4
```

### JSON Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "console"
  },
  "metrics": {
    "enabled": true,
    "port": 9090
  },
  "database": {
    "host": "localhost",
    "port": 5432,
    "pool_size": 10
  }
}
```

### TOML Configuration

```toml
# config/base.toml
[logging]
level = "INFO"
format = "console"

[metrics]
enabled = true
port = 9090

[database]
host = "localhost"
port = 5432
pool_size = 10
```

### File Priority

When loading from a directory:

1. `base.{yaml|json|toml}` - Base configuration (priority 10)
2. `{environment}.{yaml|json|toml}` - Environment-specific (priority 20)
3. `local.{yaml|json|toml}` - Local overrides (priority 30)

Example directory structure:

```
config/
├── base.yaml          # Base config (priority 10)
├── development.yaml   # Dev overrides (priority 20)
├── staging.yaml       # Staging overrides (priority 20)
├── production.yaml    # Production overrides (priority 20)
└── local.yaml         # Local overrides, not committed (priority 30)
```

---

## Best Practices

### 1. Use Environment-Specific Configuration

```yaml
# config/base.yaml - shared defaults
logging:
  level: INFO

# config/production.yaml - production overrides
logging:
  level: WARNING
  format: json

# config/development.yaml - development overrides
logging:
  level: DEBUG
```

### 2. Keep Secrets in Environment Variables or Secret Managers

```yaml
# Bad: hardcoded secrets
database:
  password: "my-secret-password"

# Good: environment variable reference
database:
  password: ${DATABASE_PASSWORD}
```

### 3. Use Configuration Presets

```python
# Use presets for common patterns
from truthound.common.resilience import CircuitBreakerConfig

# Instead of manually configuring
config = CircuitBreakerConfig.for_external_api()
```

### 4. Validate Configuration Early

```python
from truthound.infrastructure.config import ConfigManager, FileConfigSource

manager = ConfigManager()
manager.add_source(FileConfigSource("config.yaml"))

# Validate on startup
errors = manager.validate()
if errors:
    raise ValueError(f"Configuration errors: {errors}")
```

### 5. Use Type-Safe Access

```python
config = manager.config  # Access via property

# Type-safe access with defaults
timeout = config.get_int("validation.timeout", default=300)
enabled = config.get_bool("metrics.enabled", default=True)
hosts = config.get_list("database.hosts", default=["localhost"])
```

### 6. Enable Hot Reload for Development

```python
manager.add_source(FileConfigSource(
    path="config.yaml",
    watch=True,  # Enable file watching for hot reload
))
```

### 7. Use Observability Presets

```python
from truthound.stores.observability import ObservabilityConfig

# Development: minimal overhead
config = ObservabilityConfig.minimal()

# Production: full observability
config = ObservabilityConfig.production()
```

---

## See Also

- [CI/CD Integration](ci-cd.md) - Checkpoint configuration in CI/CD
- [Data Sources](datasources.md) - DataSource configuration details
- [Storage Backends](stores.md) - Store configuration options
- [Data Profiling](profiler.md) - Profiler configuration
- [Performance](performance.md) - Performance tuning configuration
