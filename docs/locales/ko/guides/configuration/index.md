# 설정 Guide

실무 운영 가이드에서 Truthound, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
from truthound.infrastructure.config import load_config

# Load configuration (auto-detects environment)
config = load_config()

# Access values with type safety
db_host = config.get("database.host", default="localhost")
db_port = config.get_int("database.port", default=5432)
debug = config.get_bool("debug", default=False)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 워크플로우s

### 워크플로우 1: Production Environment Setup

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

### 워크플로우 2: Structured 로깅 Setup

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

### 워크플로우 3: Prometheus 메트릭

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

### 워크플로우 4: 복원력 Patterns

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

### 워크플로우 5: 감사 로깅 for Compliance

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Full Documentation

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 아키텍처

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

## 설정 Priority

실무 운영 가이드에서 Configuration, Higher을(를) 다루는 항목입니다:

| 소스 | 실무 운영 가이드에서 Priority을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|----------|-------------|
| Base 파일 (`base.yaml`) | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Common 설정 |
| Environment 파일 (`production.yaml`) | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Environment-specific을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Local 파일 (`local.yaml`) | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Environment을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 런타임 overrides |
| 실무 운영 가이드에서 HashiCorp, Vault을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿 (highest priority) |
| AWS 시크릿 Manager | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 시크릿 (highest priority) |

## 빠른 시작

### Basic 설정 Loading

```python
from truthound.infrastructure.config import load_config, get_config

# Load configuration (uses defaults + environment variables)
config = load_config()

# Access configuration values
db_host = config.get("database.host", default="localhost")
db_port = config.get_int("database.port", default=5432)
debug = config.get_bool("debug", default=False)
```

### Full 설정 with All Sources

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

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

1. 실무 운영 가이드에서 `TRUTHOUND_ENV`, TRUTHOUND_ENV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 `ENVIRONMENT`, ENVIRONMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `ENV`, ENV을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `DEVELOPMENT`, DEVELOPMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Environment Aliases

| 실무 운영 가이드에서 Input을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Environment을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `dev`, `development`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 DEVELOPMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `test`, `testing`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 TESTING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `stage`, `staging`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 STAGING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `prod`, `production`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PRODUCTION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `local`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 LOCAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

## Typed 설정 Access

실무 운영 가이드에서 `ConfigProfile`, ConfigProfile을(를) 다루는 항목입니다:

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

## 설정 검증

Define a 스키마 to validate 설정:

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

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|-------------|
| 실무 운영 가이드에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설정 key (dot-separated) |
| 실무 운영 가이드에서 `type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Expected, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `required`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Whether을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `default`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `min_value`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_value`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `pattern`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Regex을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `choices`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Allowed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `description`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Default 스키마

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

```python
from truthound.infrastructure.config import create_default_schema

schema = create_default_schema()
```

### Default Fields

| 실무 운영 가이드에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Constraints을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|------|---------|-------------|
| 실무 운영 가이드에서 `logging.level`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"INFO"`, INFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `logging.format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"console"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `metrics.enabled`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `metrics.port`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `9090`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `database.host`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"localhost"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `database.port`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `5432`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `database.pool_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `10`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `validation.timeout`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `300`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `validation.max_workers`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `4`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Hot Reload

실무 운영 가이드에서 Configuration을(를) 다루는 항목입니다:

```python
from truthound.infrastructure.config import load_config, reload_config

# Enable auto-reload
config = load_config(auto_reload=True)

# Manual reload
config = reload_config()
```

### Reload Callbacks

실무 운영 가이드에서 Register을(를) 다루는 항목입니다:

```python
from truthound.infrastructure.config import ConfigManager

manager = ConfigManager(auto_reload=True)

def on_config_change(new_config):
    print("Configuration changed!")
    # Re-initialize services with new config

manager.on_reload(on_config_change)
config = manager.load()
```

## 파일 Structure

Recommended 설정 파일 structure:

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

## 설정 Guide Pages

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Configuration, Sources, File, Environment, Variables, Vault, AWS, Secrets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Resilience, Patterns, Circuit, Breaker, Retry, Bulkhead, Rate, Limiter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Logging, Configuration, Enterprise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- **[메트릭 설정](metrics.md)** - Prometheus metrics
- 실무 운영 가이드에서 Audit, Logging, Compliance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Encryption, Data, Cloud, KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- **[환경 변수](environment-vars.md)** - Complete reference

## 함께 보기

- [CI/CD 통합](../ci-cd.md) - 체크포인트 설정 in CI/CD
- 실무 운영 가이드에서 Data, Sources, DataSource을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Storage, Backends, Store을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [데이터 프로파일링](../profiler/index.md) - 프로파일러 설정
