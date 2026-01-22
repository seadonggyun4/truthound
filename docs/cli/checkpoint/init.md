# truthound checkpoint init

Initialize a sample checkpoint configuration file. This command creates a starter configuration for quick setup.

## Synopsis

```bash
truthound checkpoint init [OPTIONS]
```

## Arguments

This command has no required arguments.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `truthound.yaml` | Output file path |
| `--format` | `-f` | `yaml` | Configuration format (yaml, json) |

## Description

The `checkpoint init` command generates a sample configuration file:

1. **Creates** a well-documented configuration template
2. **Includes** common validator examples with configurations
3. **Shows** action configuration patterns (store, docs, slack)
4. **Provides** trigger configuration examples (schedule, cron)

## Examples

### Basic Initialization

```bash
truthound checkpoint init
```

Output:
```
Sample checkpoint config created: truthound.yaml

Edit the file to configure your checkpoints, then run:
  truthound checkpoint run <checkpoint_name> --config truthound.yaml
```

### Custom Output Path

```bash
truthound checkpoint init -o config/data-quality.yaml
```

### JSON Format

```bash
truthound checkpoint init --format json -o truthound.json
```

## Generated Configuration

### YAML Output (default)

```yaml
checkpoints:
- name: daily_data_validation
  data_source: data/production.csv
  validators:
  - 'null'
  - duplicate
  - range
  validator_config:
    range:
      # Column-specific range constraints
      columns:
        age:
          min_value: 0
          max_value: 150
        price:
          min_value: 0
  # Note: For regex validation, use th.check() with RegexValidator directly:
  #   RegexValidator(pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$", columns=["email"])
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
    run_on_weekdays:
    - 0
    - 1
    - 2
    - 3
    - 4

- name: hourly_metrics_check
  data_source: data/metrics.parquet
  validators:
  - 'null'
  - range
  validator_config:
    range:
      columns:
        value:
          min_value: 0
          max_value: 100
        count:
          min_value: 0
  actions:
  - type: webhook
    url: https://api.example.com/data-quality/events
    auth_type: bearer
    auth_credentials:
      token: ${API_TOKEN}
  triggers:
  - type: cron
    expression: 0 * * * *
```

### JSON Output

```json
{
  "checkpoints": [
    {
      "name": "daily_data_validation",
      "data_source": "data/production.csv",
      "validators": ["null", "duplicate", "range"],
      "validator_config": {
        "range": {
          "columns": {
            "age": {"min_value": 0, "max_value": 150},
            "price": {"min_value": 0}
          }
        }
      },
      "min_severity": "medium",
      "auto_schema": true,
      "tags": {
        "environment": "production",
        "team": "data-platform"
      },
      "actions": [
        {
          "type": "store_result",
          "store_path": "./truthound_results",
          "partition_by": "date"
        },
        {
          "type": "update_docs",
          "site_path": "./truthound_docs",
          "include_history": true
        },
        {
          "type": "slack",
          "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
          "notify_on": "failure",
          "channel": "#data-quality"
        }
      ],
      "triggers": [
        {
          "type": "schedule",
          "interval_hours": 24,
          "run_on_weekdays": [0, 1, 2, 3, 4]
        }
      ]
    },
    {
      "name": "hourly_metrics_check",
      "data_source": "data/metrics.parquet",
      "validators": ["null", "range"],
      "validator_config": {
        "range": {
          "columns": {
            "value": {"min_value": 0, "max_value": 100},
            "count": {"min_value": 0}
          }
        }
      },
      "actions": [
        {
          "type": "webhook",
          "url": "https://api.example.com/data-quality/events",
          "auth_type": "bearer",
          "auth_credentials": {"token": "${API_TOKEN}"}
        }
      ],
      "triggers": [
        {
          "type": "cron",
          "expression": "0 * * * *"
        }
      ]
    }
  ]
}
```

## Configuration Sections

### Checkpoint Definition

```yaml
checkpoints:
- name: my_checkpoint           # Required: unique identifier
  data_source: path/to/file.csv # Required: data file path
  validators:                   # Required: list of validators
  - 'null'
  - duplicate
```

### Validators

```yaml
validators:
- 'null'        # Check for null values
- duplicate     # Check for duplicates
- range         # Check numeric ranges
```

!!! note "Regex Validation"
    RegexValidator requires a single `pattern` parameter and is not configurable via YAML checkpoint config.
    Use Python API directly:
    ```python
    from truthound.validators import RegexValidator
    th.check(data, validators=[RegexValidator(pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$", columns=["email"])])
    ```

### Validator Configuration

```yaml
validator_config:
  range:
    columns:
      age:
        min_value: 0
        max_value: 150
```

### Actions

```yaml
actions:
- type: store_result          # Store validation results
  store_path: ./results
  partition_by: date

- type: update_docs           # Generate HTML documentation
  site_path: ./docs
  include_history: true

- type: slack                 # Slack notification
  webhook_url: ${SLACK_WEBHOOK_URL}
  notify_on: failure
  channel: '#data-quality'

- type: webhook               # Generic webhook
  url: https://api.example.com/webhook
  auth_type: bearer
  auth_credentials:
    token: ${API_TOKEN}
```

### Triggers

```yaml
triggers:
- type: schedule              # Time-interval based
  interval_hours: 24
  run_on_weekdays: [0, 1, 2, 3, 4]

- type: cron                  # Cron expression
  expression: "0 * * * *"     # Every hour
```

## Use Cases

### 1. New Project Setup

```bash
# Initialize new project
mkdir my-data-project && cd my-data-project
truthound checkpoint init
```

### 2. Quick Start

```bash
# Initialize, validate, and run
truthound checkpoint init

# Edit truthound.yaml to configure your data source...

# Run a checkpoint
truthound checkpoint run daily_data_validation --config truthound.yaml
```

### 3. CI/CD Template

```bash
# Generate CI/CD-ready configuration
truthound checkpoint init -o .github/truthound.yaml
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (file write error, permission denied, or other error) |

## Related Commands

- [`checkpoint validate`](validate.md) - Validate configuration
- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint list`](list.md) - List checkpoints

## See Also

- [Getting Started Guide](../../getting-started/quickstart.md)
- [CI/CD Integration](../../guides/ci-cd.md)
- [Configuration Reference](../../guides/configuration.md)
