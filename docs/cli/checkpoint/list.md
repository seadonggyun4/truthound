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
| `--config` | `-c` | None | Checkpoint configuration file |
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
truthound checkpoint list --config truthound.yaml
```

Output:
```
Checkpoints (2):
  - daily_data_validation
      Data: data/production.csv
      Actions: 3
      Triggers: 1
  - hourly_metrics_check
      Data: data/metrics.parquet
      Actions: 1
      Triggers: 1
```

### Custom Configuration File

```bash
truthound checkpoint list --config production.yaml
```

### JSON Output

```bash
truthound checkpoint list --config truthound.yaml --format json
```

Output:
```json
[
  {
    "name": "daily_data_validation",
    "config": {
      "data_source": "data/production.csv",
      "validators": ["null", "duplicate", "range", "regex"]
    },
    "actions": [...],
    "triggers": [...]
  },
  {
    "name": "hourly_metrics_check",
    "config": {
      "data_source": "data/metrics.parquet",
      "validators": ["null", "range"]
    },
    "actions": [...],
    "triggers": [...]
  }
]
```

Note: JSON output returns an array of checkpoint objects directly (not wrapped in an outer object).

### Detailed View

The console output shows a summary. For detailed checkpoint information, use JSON format and pipe to `jq`:

```bash
truthound checkpoint list --format json | jq '.[] | select(.name == "daily_data_validation")'
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
for checkpoint in $(truthound checkpoint list --format json | jq -r '.[].name'); do
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
| 1 | Error (configuration file not found, invalid, or other error) |

## Related Commands

- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint validate`](validate.md) - Validate configuration
- [`checkpoint init`](init.md) - Initialize configuration

## See Also

- [CI/CD Integration Guide](../../guides/ci-cd.md)
