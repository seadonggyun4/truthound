# Configuration Sources

Truthound supports multiple configuration sources that can be combined with priority-based merging.

## Overview

| Source Class | Description | Default Priority |
|--------------|-------------|------------------|
| `FileConfigSource` | YAML, JSON, TOML files | 10-30 |
| `EnvConfigSource` | Environment variables | 100 |
| `VaultConfigSource` | HashiCorp Vault KV v2 | 200 |
| `AwsSecretsSource` | AWS Secrets Manager | 200 |

## FileConfigSource

Load configuration from YAML, JSON, or TOML files.

### Basic Usage

```python
from truthound.infrastructure.config import ConfigManager, FileConfigSource

manager = ConfigManager()

# Add file source
manager.add_source(FileConfigSource(
    path="config/base.yaml",
    required=True,
    priority=10,
    watch=True,  # Enable file watching for hot reload
))

config = manager.load()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to configuration file |
| `required` | `bool` | `False` | Raise error if file not found |
| `priority` | `int` | `50` | Source priority (higher overrides lower) |
| `watch` | `bool` | `False` | Enable file watching for hot reload |

### Supported Formats

| Extension | Format | Dependency |
|-----------|--------|------------|
| `.yaml`, `.yml` | YAML | `pyyaml` |
| `.json` | JSON | Built-in |
| `.toml` | TOML | `tomllib` (Python 3.11+) or `tomli` |

### File Loading Behavior

- File format is auto-detected from extension
- Missing optional files return empty dict `{}`
- Missing required files raise `ConfigSourceError`
- File modification time is tracked for hot reload

### Hot Reload Support

```python
source = FileConfigSource(
    path="config.yaml",
    watch=True,
)

# Later, check for changes
new_config = source.reload()  # Only reloads if file changed
```

## EnvConfigSource

Load configuration from environment variables with automatic type inference.

### Basic Usage

```python
from truthound.infrastructure.config import EnvConfigSource

manager.add_source(EnvConfigSource(
    prefix="TRUTHOUND",
    separator="_",
    priority=100,
))
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | `"TRUTHOUND"` | Environment variable prefix |
| `separator` | `str` | `"_"` | Separator for nested keys |
| `priority` | `int` | `100` | Source priority |

### Environment Variable Mapping

Environment variables are converted to nested dictionaries:

```bash
# Environment variables
TRUTHOUND_DATABASE_HOST=localhost
TRUTHOUND_DATABASE_PORT=5432
TRUTHOUND_LOGGING_LEVEL=DEBUG
```

Becomes:

```python
{
    "database": {
        "host": "localhost",
        "port": 5432,
    },
    "logging": {
        "level": "DEBUG",
    },
}
```

### Automatic Type Inference

Values are automatically converted to appropriate types:

| Value | Converted Type | Example |
|-------|----------------|---------|
| `true`, `yes`, `1`, `on` | `bool` (True) | `TRUTHOUND_DEBUG=true` → `True` |
| `false`, `no`, `0`, `off` | `bool` (False) | `TRUTHOUND_DEBUG=false` → `False` |
| `null`, `none`, `` | `None` | `TRUTHOUND_VALUE=null` → `None` |
| Integer string | `int` | `TRUTHOUND_PORT=5432` → `5432` |
| Float string | `float` | `TRUTHOUND_TIMEOUT=30.5` → `30.5` |
| JSON array/object | `list`/`dict` | `TRUTHOUND_HOSTS=["a","b"]` → `["a", "b"]` |
| Other | `str` | `TRUTHOUND_NAME=app` → `"app"` |

## VaultConfigSource

Load secrets from HashiCorp Vault KV v2 engine.

### Basic Usage

```python
from truthound.infrastructure.config import VaultConfigSource

manager.add_source(VaultConfigSource(
    url="https://vault.example.com",
    path="truthound/config",
    token=None,  # Uses VAULT_TOKEN env var
    mount_point="secret",
    priority=200,
))
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | required | Vault server URL |
| `path` | `str` | required | Secret path |
| `token` | `str \| None` | `None` | Vault token (falls back to `VAULT_TOKEN` env var) |
| `role` | `str \| None` | `None` | AppRole for authentication |
| `mount_point` | `str` | `"secret"` | KV mount point |
| `priority` | `int` | `200` | Source priority |

### Vault Path Structure

For KV v2 engine, the URL is constructed as:

```
{url}/v1/{mount_point}/data/{path}
```

Example:
- URL: `https://vault.example.com`
- Mount Point: `secret`
- Path: `truthound/production`
- Full URL: `https://vault.example.com/v1/secret/data/truthound/production`

### Writing Secrets to Vault

```bash
# Write secrets to Vault
vault kv put secret/truthound/production \
    database_host=prod-db.example.com \
    database_password=secret123 \
    api_keys='{"slack":"xoxb-...","pagerduty":"..."}'
```

### Authentication

The source supports token-based authentication:

```python
# Explicit token
VaultConfigSource(
    url="https://vault.example.com",
    path="truthound/config",
    token="hvs.CAESIE...",
)

# Token from environment variable
VaultConfigSource(
    url="https://vault.example.com",
    path="truthound/config",
    # token defaults to os.getenv("VAULT_TOKEN")
)
```

### Hot Reload Support

```python
source = VaultConfigSource(...)
source.supports_reload  # True

# Reload from Vault
new_secrets = source.reload()
```

## AwsSecretsSource

Load secrets from AWS Secrets Manager.

### Basic Usage

```python
from truthound.infrastructure.config import AwsSecretsSource

manager.add_source(AwsSecretsSource(
    secret_name="truthound/production",
    region="us-east-1",  # Or uses AWS_REGION env var
    priority=200,
))
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `secret_name` | `str` | required | Secret name or ARN |
| `region` | `str \| None` | `None` | AWS region (falls back to `AWS_REGION` env var or `us-east-1`) |
| `priority` | `int` | `200` | Source priority |

### Secret Format

Secrets should be stored as JSON:

```json
{
    "database_host": "prod-db.amazonaws.com",
    "database_password": "secret123",
    "api_keys": {
        "slack": "xoxb-...",
        "pagerduty": "..."
    }
}
```

### Creating Secrets in AWS

```bash
# Using AWS CLI
aws secretsmanager create-secret \
    --name truthound/production \
    --secret-string '{"database_host":"prod-db.example.com","database_password":"secret123"}'

# Or update existing secret
aws secretsmanager put-secret-value \
    --secret-id truthound/production \
    --secret-string '{"database_host":"prod-db.example.com","database_password":"secret123"}'
```

### Authentication

Uses the default AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. Shared credential file (`~/.aws/credentials`)
3. IAM role (for EC2/ECS/Lambda)

### Hot Reload Support

```python
source = AwsSecretsSource(...)
source.supports_reload  # True

# Reload from AWS
new_secrets = source.reload()
```

## Combining Multiple Sources

Sources are processed in priority order (lowest first), with higher priority sources overriding values from lower priority sources.

### Example: Production Setup

```python
from truthound.infrastructure.config import (
    ConfigManager,
    Environment,
    FileConfigSource,
    EnvConfigSource,
    VaultConfigSource,
)

manager = ConfigManager(environment=Environment.PRODUCTION)

# Base config (priority 10)
manager.add_source(FileConfigSource("config/base.yaml", priority=10))

# Environment-specific config (priority 20)
manager.add_source(FileConfigSource("config/production.yaml", priority=20))

# Environment variables (priority 100)
manager.add_source(EnvConfigSource(prefix="TRUTHOUND", priority=100))

# Vault secrets (priority 200)
manager.add_source(VaultConfigSource(
    url="https://vault.example.com",
    path="truthound/production",
    priority=200,
))

config = manager.load()
```

### Merge Order

1. Load `config/base.yaml` (priority 10)
2. Merge `config/production.yaml` on top (priority 20)
3. Merge environment variables on top (priority 100)
4. Merge Vault secrets on top (priority 200)

### Using load_config Helper

The `load_config` function simplifies common setups:

```python
from truthound.infrastructure.config import load_config

config = load_config(
    environment="production",
    config_path="config/",
    env_prefix="TRUTHOUND",
    use_vault=True,
    vault_url="https://vault.example.com",
    vault_path="truthound/production",
    use_aws_secrets=False,
    auto_reload=True,
    validate=True,
)
```

This automatically sets up:

- `base.{yaml|yml|json|toml}` (priority 10)
- `{environment}.{yaml|yml|json|toml}` (priority 20)
- `local.{yaml|yml|json|toml}` (priority 30)
- Environment variables (priority 100)
- Vault/AWS secrets (priority 200)

## Creating Custom Sources

Implement the `ConfigSource` abstract base class:

```python
from truthound.infrastructure.config import ConfigSource
from typing import Any

class CustomConfigSource(ConfigSource):
    def __init__(self, priority: int = 50) -> None:
        super().__init__(priority)

    def load(self) -> dict[str, Any]:
        """Load configuration from custom source."""
        # Implement your loading logic
        return {"key": "value"}

    def reload(self) -> dict[str, Any]:
        """Reload configuration (optional)."""
        return self.load()

    @property
    def supports_reload(self) -> bool:
        """Whether this source supports hot reload."""
        return True

# Use custom source
manager.add_source(CustomConfigSource(priority=150))
```

## File Format Examples

### YAML Configuration

```yaml
# config/base.yaml
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

### JSON Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "console"
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
[logging]
level = "INFO"
format = "console"

[database]
host = "localhost"
port = 5432
pool_size = 10
```
