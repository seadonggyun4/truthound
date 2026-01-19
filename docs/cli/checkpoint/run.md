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
| `--config` | `-c` | `truthound.yaml` | Checkpoint configuration file (YAML/JSON) |
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
truthound checkpoint run daily_validation --config truthound.yaml
```

Output:
```
Checkpoint: daily_validation
============================
Running validators on 2 data assets...

Asset: customers.csv
  ✓ not_null (id, email)
  ✓ unique (id)
  ✗ range (age): 5 values outside [0, 150]

Asset: orders.csv
  ✓ not_null (order_id, customer_id)
  ✓ foreign_key (customer_id)

Summary:
  Total Assets: 2
  Total Validators: 5
  Passed: 4
  Failed: 1
  Status: FAILED
```

### Strict Mode (CI/CD)

Exit with code 1 if any issues are found:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --strict
```

### Override Data Source

Run checkpoint with different data:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --data /path/to/new/data.csv
```

### Override Validators

Run only specific validators:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --validators null,unique,range
```

### JSON Output

```bash
truthound checkpoint run daily_validation -c truthound.yaml --format json -o results.json
```

Output file (`results.json`):
```json
{
  "checkpoint": "daily_validation",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "failed",
  "assets": [
    {
      "name": "customers.csv",
      "path": "data/customers.csv",
      "rows": 10000,
      "results": [
        {
          "validator": "not_null",
          "columns": ["id", "email"],
          "passed": true
        },
        {
          "validator": "range",
          "column": "age",
          "passed": false,
          "issues": [
            {
              "severity": "high",
              "message": "5 values outside range [0, 150]"
            }
          ]
        }
      ]
    }
  ],
  "summary": {
    "total_assets": 2,
    "total_validators": 5,
    "passed": 4,
    "failed": 1
  }
}
```

### Store Results

Save results to a directory:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --store .truthound/results
```

Results are stored as:
```
.truthound/results/
└── daily_validation/
    └── 2024-01-15T10-30-00/
        ├── report.json
        └── summary.txt
```

### Slack Notification

Send results to Slack:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --slack $SLACK_WEBHOOK_URL
```

Or configure in YAML:
```yaml
checkpoints:
  daily_validation:
    notifications:
      slack:
        webhook_url: ${SLACK_WEBHOOK_URL}
        on_failure: true
        on_success: false
```

### Webhook Notification

Send results to a webhook:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --webhook https://api.example.com/webhook
```

### GitHub Actions Summary

Write summary to GitHub Actions:

```bash
truthound checkpoint run daily_validation -c truthound.yaml --github-summary
```

This creates a summary visible in the GitHub Actions job:

```markdown
## Data Quality Report: daily_validation

| Asset | Validators | Passed | Failed |
|-------|------------|--------|--------|
| customers.csv | 3 | 2 | 1 |
| orders.csv | 2 | 2 | 0 |

### Issues Found

- **customers.csv**: range (age) - 5 values outside range [0, 150]
```

## Configuration File

### Minimal Configuration

```yaml
checkpoints:
  my_checkpoint:
    data_assets:
      - name: data
        path: data.csv
    validators:
      - type: not_null
        columns: [id]
```

### Full Configuration

```yaml
checkpoints:
  daily_validation:
    description: "Daily data quality validation"
    schedule: "0 6 * * *"  # Cron expression (informational)

    data_assets:
      - name: customers
        path: data/customers.csv
        format: csv
      - name: orders
        path: data/orders.parquet
        format: parquet

    validators:
      - type: not_null
        columns: [id, email]
        severity: critical

      - type: unique
        columns: [id]
        severity: critical

      - type: range
        column: age
        min_value: 0
        max_value: 150
        severity: high

      - type: pattern
        column: email
        pattern: email
        severity: medium

      - type: allowed_values
        column: status
        values: [active, inactive, pending]

    notifications:
      slack:
        webhook_url: ${SLACK_WEBHOOK_URL}
        channel: "#data-quality"
        on_failure: true
        on_success: false
        mention_on_failure: "@here"

      webhook:
        url: https://api.example.com/webhook
        method: POST
        headers:
          Content-Type: application/json
          Authorization: Bearer ${API_TOKEN}
        on_failure: true

    store:
      path: .truthound/results
      retention_days: 30
      format: json
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success (no issues, or issues found without `--strict`) |
| 1 | Issues found with `--strict` flag |
| 2 | Configuration error or invalid arguments |

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
    truthound checkpoint run daily_validation \
      --config truthound.yaml \
      --strict \
      --github-summary
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 2. Scheduled Validation

```bash
# Cron job
0 6 * * * truthound checkpoint run daily_validation -c /app/truthound.yaml --store /var/log/truthound
```

### 3. Pre-Deployment Check

```bash
# Before deployment
truthound checkpoint run pre_deploy -c truthound.yaml --strict || exit 1
```

### 4. Multiple Checkpoints

```bash
# Run multiple checkpoints
for checkpoint in daily_validation weekly_drift monthly_audit; do
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
