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

Name                      Data Source             Validators    Actions
─────────────────────────────────────────────────────────────────────────────
daily_data_validation     data/production.csv     4             store, docs, slack
hourly_metrics_check      data/metrics.parquet    2             webhook

Total: 2 checkpoints
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
      "name": "daily_data_validation",
      "data_source": "data/production.csv",
      "validators": ["null", "duplicate", "range", "regex"],
      "validator_count": 4,
      "actions": ["store_result", "update_docs", "slack"],
      "tags": {
        "environment": "production",
        "team": "data-platform"
      }
    },
    {
      "name": "hourly_metrics_check",
      "data_source": "data/metrics.parquet",
      "validators": ["null", "range"],
      "validator_count": 2,
      "actions": ["webhook"]
    }
  ],
  "total": 2
}
```

### Detailed View

The console output shows a summary. For detailed checkpoint information, use JSON format and pipe to `jq`:

```bash
truthound checkpoint list --format json | jq '.checkpoints[] | select(.name == "daily_data_validation")'
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
