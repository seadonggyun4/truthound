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
2. **Includes** common validator examples
3. **Shows** notification configuration patterns
4. **Provides** storage configuration examples

## Examples

### Basic Initialization

```bash
truthound checkpoint init
```

Output:
```
Created: truthound.yaml

Next steps:
  1. Edit truthound.yaml to configure your checkpoints
  2. Validate: truthound checkpoint validate truthound.yaml
  3. Run: truthound checkpoint run <checkpoint_name>
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
# Truthound Checkpoint Configuration
# Documentation: https://truthound.io/docs/cli/checkpoint/

name: my_data_quality_pipeline
version: "1.0"

# Define your checkpoints
checkpoints:
  # Example: Daily data validation
  daily_validation:
    description: "Daily data quality check"

    # Data assets to validate
    data_assets:
      - name: customers
        path: data/customers.csv
        # format: csv  # auto-detected from extension

      - name: orders
        path: data/orders.csv

    # Validation rules
    validators:
      # Check for null values in critical columns
      - type: not_null
        columns: [id, email]
        severity: critical

      # Ensure ID uniqueness
      - type: unique
        columns: [id]
        severity: critical

      # Validate age range
      - type: range
        column: age
        min_value: 0
        max_value: 150
        severity: high

      # Validate email format
      - type: pattern
        column: email
        pattern: email
        severity: medium

      # Check allowed status values
      - type: allowed_values
        column: status
        values: [active, inactive, pending]
        severity: medium

    # Notification settings (optional)
    notifications:
      # Slack notifications
      slack:
        webhook_url: ${SLACK_WEBHOOK_URL}  # Use environment variable
        on_failure: true
        on_success: false

      # Generic webhook
      # webhook:
      #   url: https://api.example.com/webhook
      #   on_failure: true

    # Result storage (optional)
    store:
      path: .truthound/results
      retention_days: 30

  # Example: Weekly drift detection
  # weekly_drift_check:
  #   description: "Weekly drift detection"
  #   data_assets:
  #     - name: baseline
  #       path: baseline/data.csv
  #     - name: current
  #       path: data/current.csv
  #   compare:
  #     baseline: baseline
  #     current: current
  #     method: psi
  #     threshold: 0.1

# Global settings (optional)
# settings:
#   default_severity: medium
#   fail_on_warning: false
#   parallel_execution: true
```

### JSON Output

```json
{
  "name": "my_data_quality_pipeline",
  "version": "1.0",
  "checkpoints": {
    "daily_validation": {
      "description": "Daily data quality check",
      "data_assets": [
        {
          "name": "customers",
          "path": "data/customers.csv"
        },
        {
          "name": "orders",
          "path": "data/orders.csv"
        }
      ],
      "validators": [
        {
          "type": "not_null",
          "columns": ["id", "email"],
          "severity": "critical"
        },
        {
          "type": "unique",
          "columns": ["id"],
          "severity": "critical"
        },
        {
          "type": "range",
          "column": "age",
          "min_value": 0,
          "max_value": 150,
          "severity": "high"
        }
      ],
      "notifications": {
        "slack": {
          "webhook_url": "${SLACK_WEBHOOK_URL}",
          "on_failure": true,
          "on_success": false
        }
      },
      "store": {
        "path": ".truthound/results",
        "retention_days": 30
      }
    }
  }
}
```

## Configuration Sections

### Data Assets

```yaml
data_assets:
  - name: unique_name        # Required: identifier
    path: path/to/file.csv   # Required: file path
    format: csv              # Optional: csv, json, parquet, ndjson
```

### Validators

```yaml
validators:
  - type: validator_type     # Required: validator type
    columns: [col1, col2]    # Required for multi-column validators
    column: col1             # Required for single-column validators
    severity: high           # Optional: low, medium, high, critical
```

### Notifications

```yaml
notifications:
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}
    on_failure: true
    on_success: false

  webhook:
    url: https://example.com/hook
    headers:
      Authorization: Bearer ${TOKEN}
```

### Storage

```yaml
store:
  path: .truthound/results
  retention_days: 30
  format: json
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
# Edit truthound.yaml...
truthound checkpoint validate truthound.yaml
truthound checkpoint run daily_validation
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
| 1 | Output file already exists (use different path) |
| 2 | Permission denied |

## Related Commands

- [`checkpoint validate`](validate.md) - Validate configuration
- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint list`](list.md) - List checkpoints

## See Also

- [Getting Started Guide](../../getting-started/quickstart.md)
- [CI/CD Integration](../../guides/ci-cd.md)
- [Configuration Reference](../../guides/configuration.md)
