---
title: Mage io_config.yaml
---

# Mage `io_config.yaml`

Truthound can read Mage's `io_config.yaml` or `io_config.yml` so that data source
details live with the Mage project instead of being duplicated in every quality block.

## Discovery Behavior

The loader searches the current directory and parent directories for:

- `io_config.yaml`
- `io_config.yml`

The lookup is read-only. If no config file is found, Truthound still falls back to its
safe runtime defaults.

## Example Shape

```yaml
default:
  TRUTHOUND_ENGINE: truthound
  TRUTHOUND_TIMEOUT: 300

data_sources:
  warehouse:
    type: postgres
    host: localhost
    port: 5432
    database: analytics
    username: ${PGUSER}
    password: ${PGPASSWORD}

  bronze_users:
    type: file
    path: data/bronze/users.parquet
    format: parquet
```

## Supported Source Families

The config layer supports common source families such as:

- file-backed inputs
- PostgreSQL and MySQL
- Snowflake, BigQuery, and Redshift
- S3, GCS, and Azure-style object stores
- custom source definitions with extra options

## Environment Variable Resolution

Environment placeholders such as `${PGPASSWORD}` are resolved when config values are
read. This keeps secrets out of checked-in project files.

## How To Use It

```python
from truthound_mage import load_io_config

io_config = load_io_config()
source = io_config.get_source("warehouse")
```

In practice, many teams will not call `load_io_config()` directly in every block. They
centralize it once in a helper file and pass the resulting source configuration into
block factories.

## Operational Guidance

- keep source names stable across environments
- prefer environment-backed credentials over literal secrets
- treat `io_config.yaml` as project infrastructure, not ad hoc local state

## Related Pages

- [Project Layout](project-layout.md)
- [Recipes](recipes.md)
