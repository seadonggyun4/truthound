# truthound checkpoint run

Run a validation checkpoint. This command executes a named checkpoint from the configuration file.

## Synopsis

```bash
truthound checkpoint run <name> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Name of the checkpoint to run |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | None | Checkpoint configuration file (YAML/JSON) |
| `--data` | `-d` | None | Override data source path |
| `--validators` | `-v` | None | Override validators (comma-separated) |
| `--output` | `-o` | None | Output file path (JSON) |
| `--format` | `-f` | `console` | Output format (console, json) |
| `--strict` | | `false` | Exit with code 1 if issues found |
| `--store` | | None | Results storage directory |
| `--slack` | | None | Slack webhook URL |
| `--webhook` | | None | Generic webhook URL |
| `--github-summary` | | `false` | Write GitHub Actions job summary |

## Description

The `checkpoint run` command executes a named checkpoint:

1. **Loads** the checkpoint configuration
2. **Validates** the data assets
3. **Runs** all configured validators
4. **Sends** notifications (if configured)
5. **Stores** results (if configured)

## Examples

### Basic Execution

```bash
truthound checkpoint run daily_data_validation --config truthound.yaml
```

Output:
```
Checkpoint: daily_data_validation
=================================
Running validators on data/production.csv...

Validators:
  ✓ null
  ✓ duplicate
  ✗ range (age): 5 values outside [0, 150]
  ✓ regex (email)

Summary:
  Total Validators: 4
  Passed: 3
  Failed: 1
  Status: FAILED
```

### Strict Mode (CI/CD)

Exit with code 1 if any issues are found:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --strict
```

### Override Data Source

Run checkpoint with different data:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --data /path/to/new/data.csv
```

### Override Validators

Run only specific validators:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --validators null,duplicate,range
```

### JSON Output

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --format json -o results.json
```

Output file (`results.json`):
```json
{
  "checkpoint": "daily_data_validation",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "failed",
  "data_source": "data/production.csv",
  "rows": 10000,
  "results": [
    {
      "validator": "null",
      "passed": true
    },
    {
      "validator": "duplicate",
      "passed": true
    },
    {
      "validator": "range",
      "passed": false,
      "issues": [
        {
          "severity": "high",
          "message": "5 values outside range [0, 150]"
        }
      ]
    },
    {
      "validator": "regex",
      "passed": true
    }
  ],
  "summary": {
    "total_validators": 4,
    "passed": 3,
    "failed": 1
  }
}
```

### Store Results

Save results to a directory:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --store .truthound/results
```

Results are stored as:
```
.truthound/results/
└── daily_data_validation/
    └── 2024-01-15T10-30-00/
        ├── report.json
        └── summary.txt
```

### Slack Notification

Send results to Slack:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --slack $SLACK_WEBHOOK_URL
```

Or configure in YAML:
```yaml
checkpoints:
- name: daily_data_validation
  data_source: data/production.csv
  validators:
  - 'null'
  - duplicate
  actions:
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    notify_on: failure
    channel: '#data-quality'
```

### Webhook Notification

Send results to a webhook:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --webhook https://api.example.com/webhook
```

### GitHub Actions Summary

Write summary to GitHub Actions:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --github-summary
```

This creates a summary visible in the GitHub Actions job:

```markdown
## Data Quality Report: daily_data_validation

| Validator | Status |
|-----------|--------|
| null | Passed |
| duplicate | Passed |
| range | Failed |
| regex | Passed |

### Issues Found

- **range**: 5 values outside range [0, 150]
```

## Configuration File

### Minimal Configuration

```yaml
checkpoints:
- name: my_checkpoint
  data_source: data.csv
  validators:
  - 'null'
```

### Full Configuration

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
    run_on_weekdays: [0, 1, 2, 3, 4]

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

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (no issues, or issues found without `--strict`) |
| 1 | Error (issues found with `--strict` flag, configuration error, or other error) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TRUTHOUND_CONFIG` | Default config file path |
| `SLACK_WEBHOOK_URL` | Slack webhook URL |
| `WEBHOOK_URL` | Generic webhook URL |
| `API_TOKEN` | API authentication token |

## Use Cases

### 1. CI/CD Pipeline

```yaml
# GitHub Actions
- name: Run Data Quality Check
  run: |
    truthound checkpoint run daily_data_validation \
      --config truthound.yaml \
      --strict \
      --github-summary
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 2. Scheduled Validation

```bash
# Cron job
0 6 * * * truthound checkpoint run daily_data_validation -c /app/truthound.yaml --store /var/log/truthound
```

### 3. Pre-Deployment Check

```bash
# Before deployment
truthound checkpoint run daily_data_validation -c truthound.yaml --strict || exit 1
```

### 4. Multiple Checkpoints

```bash
# Run multiple checkpoints
for checkpoint in daily_data_validation hourly_metrics_check; do
  truthound checkpoint run $checkpoint -c truthound.yaml --strict
done
```

## Related Commands

- [`checkpoint list`](list.md) - List available checkpoints
- [`checkpoint validate`](validate.md) - Validate configuration
- [`checkpoint init`](init.md) - Initialize configuration
- [`check`](../core/check.md) - Single file validation

## See Also

- [CI/CD Integration Guide](../../guides/ci-cd.md)
- [Notification Configuration](../../guides/notifications.md)
- [Storage Backends](../../guides/stores.md)
