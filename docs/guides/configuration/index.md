# Configuration Guide

This guide covers configuration management with Truthound's Python API. It includes practical workflows for environment setup, secrets management, logging, metrics, and resilience patterns.

---

## Quick Start

```python
from truthound.infrastructure.config import load_config

# Load configuration (auto-detects environment)
config = load_config()

# Access values with type safety
db_host = config.get("database.host", default="localhost")
db_port = config.get_int("database.port", default=5432)
debug = config.get_bool("debug", default=False)
```

---

## Common Workflows

### Workflow 1: Production Environment Setup

```python
from truthound.infrastructure.config import load_config, Environment

# Production configuration with Vault secrets
config = load_config(
    environment=Environment.PRODUCTION,
    config_path="config/",
    env_prefix="TRUTHOUND",
    use_vault=True,
    vault_url="http://vault:8200",
    vault_path="truthound/production",
    validate=True,
)

# Access secrets (fetched from Vault)
db_password = config.get("database.password")
api_key = config.get("external_api.key")
```

### Workflow 2: Structured Logging Setup

```python
from truthound.infrastructure.logging import setup_logging, LogConfig

# Configure structured JSON logging
log_config = LogConfig(
    level="INFO",
    format="json",
    correlation_id_header="X-Request-ID",
    include_timestamp=True,
    include_hostname=True,
)

logger = setup_logging(log_config)

# Log with correlation ID
logger.info(
    "Validation completed",
    extra={
        "data_asset": "customers.csv",
        "issue_count": 5,
        "duration_ms": 1234,
    }
)
```

### Workflow 3: Prometheus Metrics

```python
from truthound.infrastructure.metrics import MetricsCollector, start_http_server

# Initialize metrics
metrics = MetricsCollector()

# Start Prometheus endpoint
start_http_server(port=9090)

# Record validation metrics
with metrics.validation_duration.time():
    report = th.check("data.csv")

metrics.validation_total.inc(labels={"status": "success"})
metrics.issues_found.inc(len(report.issues))
```

### Workflow 4: Resilience Patterns

```python
from truthound.common.resilience import ResilienceBuilder
import truthound as th

# Build resilience policy
resilience = (
    ResilienceBuilder()
    .with_retry(max_attempts=3, backoff_factor=2.0)
    .with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    .with_timeout(seconds=60)
    .with_bulkhead(max_concurrent=10)
    .build()
)

# Execute with resilience
from truthound.datasources import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="db.example.com")

result = resilience.execute(lambda: th.check(source=source))
```

### Workflow 5: Audit Logging for Compliance

```python
from truthound.infrastructure.audit import AuditLogger, AuditConfig

# Configure audit logging
audit_config = AuditConfig(
    enabled=True,
    storage="elasticsearch",
    elasticsearch_url="http://elasticsearch:9200",
    index_prefix="truthound-audit",
    retention_days=365,  # SOC 2 compliance
)

audit = AuditLogger(audit_config)

# Log validation operations
audit.log(
    operation="validation",
    actor="system",
    resource="customers.csv",
    result="success",
    metadata={"issue_count": 0},
)
```

---

## Full Documentation

Truthound provides an enterprise-grade configuration management system with support for multiple environments, configuration sources, validation, and hot reloading.

## Architecture

```
ConfigSource[] (ordered by priority)
     |
     +---> EnvConfigSource (environment variables)
     +---> FileConfigSource (YAML, JSON, TOML)
     +---> VaultConfigSource (HashiCorp Vault)
     +---> AwsSecretsSource (AWS Secrets Manager)
     |
     v
ConfigManager
     |
     +---> Merge & Validate
     |
     v
ConfigProfile (typed access)
```

## Configuration Priority

Configuration sources are processed in priority order. Higher priority sources override lower priority ones:

| Source | Priority | Description |
|--------|----------|-------------|
| Base file (`base.yaml`) | 10 | Common configuration |
| Environment file (`production.yaml`) | 20 | Environment-specific |
| Local file (`local.yaml`) | 30 | Local overrides (git-ignored) |
| Environment variables | 100 | Runtime overrides |
| HashiCorp Vault | 200 | Secrets (highest priority) |
| AWS Secrets Manager | 200 | Secrets (highest priority) |

## Quick Start

### Basic Configuration Loading

```python
from truthound.infrastructure.config import load_config, get_config

# Load configuration (uses defaults + environment variables)
config = load_config()

# Access configuration values
db_host = config.get("database.host", default="localhost")
db_port = config.get_int("database.port", default=5432)
debug = config.get_bool("debug", default=False)
```

### Full Configuration with All Sources

```python
from truthound.infrastructure.config import (
    load_config,
    Environment,
)

config = load_config(
    environment=Environment.PRODUCTION,
    config_path="config/",
    env_prefix="TRUTHOUND",
    use_vault=True,
    vault_url="http://vault:8200",
    vault_path="truthound/production",
    use_aws_secrets=True,
    aws_secret_name="truthound/production",
    auto_reload=True,
    validate=True,
)
```

## Environments

Truthound supports five environment types:

```python
from truthound.infrastructure.config import Environment

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
```

### Environment Detection

The environment is detected from environment variables in this order:

1. `TRUTHOUND_ENV`
2. `ENVIRONMENT`
3. `ENV`

If none are set, defaults to `DEVELOPMENT`.

### Environment Aliases

| Input | Environment |
|-------|-------------|
| `dev`, `development` | DEVELOPMENT |
| `test`, `testing` | TESTING |
| `stage`, `staging` | STAGING |
| `prod`, `production` | PRODUCTION |
| `local` | LOCAL |

### Environment Properties

```python
config = load_config()

# Check environment type
if config.is_production:  # True for PRODUCTION and STAGING
    # Production-specific logic
    pass

if config.is_development:  # True for DEVELOPMENT, LOCAL, and TESTING
    # Development-specific logic
    pass
```

## Typed Configuration Access

`ConfigProfile` provides type-safe access to configuration values:

```python
config = load_config()

# String (default)
host = config.get("database.host", default="localhost")
host = config.get_str("database.host", default="localhost")

# Integer
port = config.get_int("database.port", default=5432)

# Float
timeout = config.get_float("request.timeout", default=30.0)

# Boolean
debug = config.get_bool("debug", default=False)

# List
hosts = config.get_list("database.hosts", default=[])

# Dictionary
options = config.get_dict("database.options", default={})

# Required values (raises ConfigError if missing)
secret = config.get("api.secret", required=True)
```

## Configuration Validation

Define a schema to validate configuration:

```python
from truthound.infrastructure.config import (
    ConfigSchema,
    ConfigField,
    ConfigValidator,
)

# Define schema
schema = ConfigSchema()
schema.add_field("database.host", str, required=True)
schema.add_field(
    "database.port",
    int,
    default=5432,
    min_value=1,
    max_value=65535,
)
schema.add_field(
    "logging.level",
    str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
)

# Validate
validator = ConfigValidator(schema)
errors = validator.validate(config_dict)
if errors:
    raise ConfigValidationError(errors)
```

### ConfigField Options

| Option | Type | Description |
|--------|------|-------------|
| `name` | str | Configuration key (dot-separated) |
| `type` | type | Expected Python type |
| `required` | bool | Whether the field is required |
| `default` | Any | Default value if not found |
| `min_value` | float | Minimum value (for numbers) |
| `max_value` | float | Maximum value (for numbers) |
| `pattern` | str | Regex pattern (for strings) |
| `choices` | list | Allowed values |
| `description` | str | Field description |

## Default Schema

Truthound includes a default schema with common configuration fields:

```python
from truthound.infrastructure.config import create_default_schema

schema = create_default_schema()
```

### Default Fields

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `logging.level` | str | `"INFO"` | TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `logging.format` | str | `"console"` | console, json, logfmt |
| `metrics.enabled` | bool | `True` | - |
| `metrics.port` | int | `9090` | 1-65535 |
| `database.host` | str | `"localhost"` | - |
| `database.port` | int | `5432` | 1-65535 |
| `database.pool_size` | int | `10` | 1-100 |
| `validation.timeout` | int | `300` | >= 1 |
| `validation.max_workers` | int | `4` | 1-32 |

## Hot Reload

Configuration supports automatic reloading when sources change:

```python
from truthound.infrastructure.config import load_config, reload_config

# Enable auto-reload
config = load_config(auto_reload=True)

# Manual reload
config = reload_config()
```

### Reload Callbacks

Register callbacks to be notified when configuration changes:

```python
from truthound.infrastructure.config import ConfigManager

manager = ConfigManager(auto_reload=True)

def on_config_change(new_config):
    print("Configuration changed!")
    # Re-initialize services with new config

manager.on_reload(on_config_change)
config = manager.load()
```

## File Structure

Recommended configuration file structure:

```
config/
├── base.yaml        # Common configuration (priority 10)
├── development.yaml # Development-specific (priority 20)
├── staging.yaml     # Staging-specific (priority 20)
├── production.yaml  # Production-specific (priority 20)
└── local.yaml       # Local overrides, git-ignored (priority 30)
```

### Example: base.yaml

```yaml
logging:
  level: INFO
  format: console

database:
  host: localhost
  port: 5432
  pool_size: 10

validation:
  timeout: 300
  max_workers: 4

metrics:
  enabled: true
  port: 9090
```

### Example: production.yaml

```yaml
logging:
  level: WARNING
  format: json

database:
  pool_size: 50

metrics:
  enable_http: true
  push_gateway_url: http://pushgateway:9091
```

## Configuration Guide Pages

This guide is split into the following sections for detailed documentation:

- **[Configuration Sources](sources.md)** - File, Environment Variables, Vault, AWS Secrets Manager
- **[Resilience Patterns](resilience.md)** - Circuit Breaker, Retry, Bulkhead, Rate Limiter
- **[Logging Configuration](logging.md)** - Enterprise logging with multiple sinks
- **[Metrics Configuration](metrics.md)** - Prometheus metrics
- **[Audit Logging](audit.md)** - Compliance and audit trails
- **[Encryption](encryption.md)** - Data encryption and Cloud KMS
- **[Environment Variables](environment-vars.md)** - Complete reference

## See Also

- [CI/CD Integration](../ci-cd.md) - Checkpoint configuration in CI/CD
- [Data Sources](../datasources/) - DataSource configuration details
- [Storage Backends](../stores/) - Store configuration options
- [Data Profiling](../profiler/) - Profiler configuration
