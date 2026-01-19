# truthound checkpoint list

List all available checkpoints in a configuration file.

## Synopsis

```bash
truthound checkpoint list [OPTIONS]
```

## Arguments

This command has no required arguments.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | `truthound.yaml` | Checkpoint configuration file |
| `--format` | `-f` | `console` | Output format (console, json) |

## Description

The `checkpoint list` command displays all checkpoints defined in the configuration file:

1. **Lists** all checkpoint names
2. **Shows** descriptions and data assets
3. **Displays** validator counts
4. **Reports** notification settings

## Examples

### Basic Usage

```bash
truthound checkpoint list
```

Output:
```
Available Checkpoints
=====================
Config: truthound.yaml

Name                    Assets    Validators    Notifications
────────────────────────────────────────────────────────────────
daily_validation        2         5             slack, webhook
weekly_drift_check      2         1             slack
monthly_audit           4         12            email, pagerduty
schema_validation       1         3             -

Total: 4 checkpoints
```

### Custom Configuration File

```bash
truthound checkpoint list --config production.yaml
```

### JSON Output

```bash
truthound checkpoint list --format json
```

Output:
```json
{
  "config_file": "truthound.yaml",
  "checkpoints": [
    {
      "name": "daily_validation",
      "description": "Daily data quality check",
      "data_assets": [
        {"name": "customers", "path": "data/customers.csv"},
        {"name": "orders", "path": "data/orders.csv"}
      ],
      "validator_count": 5,
      "notifications": ["slack", "webhook"]
    },
    {
      "name": "weekly_drift_check",
      "description": "Weekly drift detection",
      "data_assets": [
        {"name": "baseline", "path": "baseline/data.csv"},
        {"name": "current", "path": "data/current.csv"}
      ],
      "validator_count": 1,
      "notifications": ["slack"]
    }
  ],
  "total": 2
}
```

### Detailed View

The console output shows a summary. For detailed checkpoint information, use JSON format and pipe to `jq`:

```bash
truthound checkpoint list --format json | jq '.checkpoints[] | select(.name == "daily_validation")'
```

## Use Cases

### 1. Discovery

Find available checkpoints in a project:

```bash
truthound checkpoint list -c truthound.yaml
```

### 2. CI/CD Script

List checkpoints for automated execution:

```bash
# Run all checkpoints
for checkpoint in $(truthound checkpoint list --format json | jq -r '.checkpoints[].name'); do
  truthound checkpoint run $checkpoint --strict
done
```

### 3. Documentation

Generate checkpoint documentation:

```bash
truthound checkpoint list --format json > docs/checkpoints.json
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 2 | Configuration file not found or invalid |

## Related Commands

- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint validate`](validate.md) - Validate configuration
- [`checkpoint init`](init.md) - Initialize configuration

## See Also

- [CI/CD Integration Guide](../../guides/ci-cd.md)
