# Environment Variables Reference

Complete reference for all Truthound environment variables.

## Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_ENV` | Environment (development, staging, production) | `development` |
| `ENVIRONMENT` | Alternative environment variable | - |
| `ENV` | Alternative environment variable | - |

## Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
| `LOG_FORMAT` | Log format (console, json, logfmt) | `console` |
| `SERVICE_NAME` | Service identifier | `""` |
| `LOG_INCLUDE_CALLER` | Include file:line:func in logs | `false` |

## Metrics

| Variable | Description | Default |
|----------|-------------|---------|
| `METRICS_ENABLED` | Enable metrics collection | `true` |
| `METRICS_HTTP_ENABLED` | Enable HTTP metrics endpoint | `false` |
| `METRICS_PORT` | Metrics HTTP port | `9090` |
| `METRICS_PUSH_GATEWAY_URL` | Prometheus push gateway URL | `""` |

## Database

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_DATABASE_HOST` | Database host | `localhost` |
| `TRUTHOUND_DATABASE_PORT` | Database port | `5432` |
| `TRUTHOUND_DATABASE_POOL_SIZE` | Connection pool size | `10` |

## Validation

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_VALIDATION_TIMEOUT` | Validation timeout in seconds | `300` |
| `TRUTHOUND_VALIDATION_MAX_WORKERS` | Maximum worker threads | `4` |

## Suite Generator

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_SUITE_STRICTNESS` | Suite strictness level (strict, medium, loose) | `medium` |
| `TRUTHOUND_SUITE_MIN_CONFIDENCE` | Minimum confidence level (low, medium, high) | `medium` |
| `TRUTHOUND_SUITE_FORMAT` | Output format (yaml, json, python, toml) | `yaml` |
| `TRUTHOUND_SUITE_INCLUDE_CATEGORIES` | Categories to include (comma-separated) | all |
| `TRUTHOUND_SUITE_EXCLUDE_CATEGORIES` | Categories to exclude (comma-separated) | none |

## Secret Managers

### HashiCorp Vault

| Variable | Description |
|----------|-------------|
| `VAULT_TOKEN` | Vault authentication token |
| `VAULT_ADDR` | Vault server address |

### AWS

| Variable | Description |
|----------|-------------|
| `AWS_REGION` | AWS region |
| `AWS_ACCESS_KEY_ID` | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_SESSION_TOKEN` | AWS session token (temporary credentials) |

### GCP

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key file |

### Azure

| Variable | Description |
|----------|-------------|
| `AZURE_TENANT_ID` | Azure tenant ID |
| `AZURE_CLIENT_ID` | Azure client ID |
| `AZURE_CLIENT_SECRET` | Azure client secret |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage connection string |

## Encryption

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_MASTER_KEY` | Local key provider password | - |

## Nested Configuration

Environment variables with the `TRUTHOUND_` prefix are automatically converted to nested configuration:

```bash
# Sets database.host
TRUTHOUND_DATABASE_HOST=localhost

# Sets database.port
TRUTHOUND_DATABASE_PORT=5432

# Sets logging.level
TRUTHOUND_LOGGING_LEVEL=DEBUG

# Sets validation.timeout
TRUTHOUND_VALIDATION_TIMEOUT=300

# Sets validation.max_workers
TRUTHOUND_VALIDATION_MAX_WORKERS=4
```

## Type Inference

Values are automatically converted to appropriate types:

| Value | Type | Example |
|-------|------|---------|
| `true`, `yes`, `1`, `on` | `bool` (True) | `TRUTHOUND_DEBUG=true` |
| `false`, `no`, `0`, `off` | `bool` (False) | `TRUTHOUND_DEBUG=false` |
| `null`, `none`, `` | `None` | `TRUTHOUND_VALUE=null` |
| Integer string | `int` | `TRUTHOUND_PORT=5432` |
| Float string | `float` | `TRUTHOUND_TIMEOUT=30.5` |
| JSON array/object | `list`/`dict` | `TRUTHOUND_HOSTS=["a","b"]` |
| Other | `str` | `TRUTHOUND_NAME=app` |

## Examples

### Development Setup

```bash
export TRUTHOUND_ENV=development
export LOG_LEVEL=DEBUG
export LOG_FORMAT=console
export METRICS_ENABLED=false
```

### Production Setup

```bash
export TRUTHOUND_ENV=production
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export SERVICE_NAME=truthound-api
export ENVIRONMENT=production

# Metrics
export METRICS_ENABLED=true
export METRICS_HTTP_ENABLED=true
export METRICS_PORT=9090

# Database
export TRUTHOUND_DATABASE_HOST=prod-db.example.com
export TRUTHOUND_DATABASE_PORT=5432
export TRUTHOUND_DATABASE_POOL_SIZE=50

# Validation
export TRUTHOUND_VALIDATION_TIMEOUT=600
export TRUTHOUND_VALIDATION_MAX_WORKERS=8

# AWS
export AWS_REGION=us-east-1

# Vault
export VAULT_TOKEN=hvs.CAESIE...
export VAULT_ADDR=https://vault.example.com
```

### Docker Compose

```yaml
version: '3.8'
services:
  truthound:
    image: truthound:latest
    environment:
      TRUTHOUND_ENV: production
      LOG_LEVEL: INFO
      LOG_FORMAT: json
      SERVICE_NAME: truthound
      ENVIRONMENT: production
      METRICS_ENABLED: 'true'
      METRICS_HTTP_ENABLED: 'true'
      METRICS_PORT: '9090'
      TRUTHOUND_DATABASE_HOST: postgres
      TRUTHOUND_DATABASE_PORT: '5432'
      TRUTHOUND_DATABASE_POOL_SIZE: '20'
      AWS_REGION: us-east-1
    secrets:
      - vault_token
      - aws_credentials

secrets:
  vault_token:
    external: true
  aws_credentials:
    external: true
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: truthound-config
data:
  TRUTHOUND_ENV: production
  LOG_LEVEL: INFO
  LOG_FORMAT: json
  SERVICE_NAME: truthound
  ENVIRONMENT: production
  METRICS_ENABLED: "true"
  METRICS_HTTP_ENABLED: "true"
  METRICS_PORT: "9090"
  TRUTHOUND_DATABASE_HOST: postgres-service
  TRUTHOUND_DATABASE_PORT: "5432"
  TRUTHOUND_DATABASE_POOL_SIZE: "20"
  TRUTHOUND_VALIDATION_TIMEOUT: "600"
  TRUTHOUND_VALIDATION_MAX_WORKERS: "8"
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: truthound-secrets
type: Opaque
stringData:
  VAULT_TOKEN: hvs.CAESIE...
  AWS_ACCESS_KEY_ID: AKIA...
  AWS_SECRET_ACCESS_KEY: ...
```

## Priority

When the same configuration is set in multiple places:

1. **Environment variables** (highest priority)
2. **Local config file** (`local.yaml`)
3. **Environment-specific config** (`production.yaml`)
4. **Base config file** (`base.yaml`)

Environment variables always override file-based configuration.
