# Checkpoint Basics

A Checkpoint bundles data sources, validators, and actions into a single execution unit to construct an automated data quality validation pipeline.

## Creating a Checkpoint

### Python API

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.checkpoint.actions import StoreValidationResult, SlackNotification

# Basic creation
checkpoint = Checkpoint(
    name="daily_user_validation",
    data_source="users.csv",
    validators=["null", "duplicate", "range"],
)

# Using Config object
config = CheckpointConfig(
    name="production_validation",
    data_source="s3://bucket/data.parquet",
    validators=["null", "duplicate", "range"],
    validator_config={
        "range": {"column": "age", "min": 0, "max": 120}
    },
    min_severity="medium",
    fail_on_critical=True,
    fail_on_high=False,
    timeout_seconds=3600,
    sample_size=100000,
    tags={"env": "production", "team": "data-platform"},
    metadata={"owner": "data-team@company.com"},
)

checkpoint = Checkpoint(config=config)
```

### YAML Configuration

```yaml
# truthound.yaml
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
    min_severity: medium
    auto_schema: true
    tags:
      environment: production
      team: data-platform
    actions:
      - type: store_result
        store_path: ./truthound_results
        partition_by: date
      - type: slack
        webhook_url: https://hooks.slack.com/services/...
        notify_on: failure
```

## CheckpointConfig Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `str` | `"default_checkpoint"` | Unique checkpoint name |
| `data_source` | `str \| Any` | `""` | Data source path or object |
| `validators` | `list[str \| Validator]` | `None` | List of validators to execute |
| `validator_config` | `dict` | `{}` | Per-validator configuration |
| `min_severity` | `str` | `None` | Minimum severity filter (`critical`, `high`, `medium`, `low`) |
| `schema` | `str` | `None` | Schema file path |
| `auto_schema` | `bool` | `False` | Enable automatic schema inference |
| `run_name_template` | `str` | `"%Y%m%d_%H%M%S"` | run_id generation template |
| `fail_on_critical` | `bool` | `True` | Treat as failure on critical issues |
| `fail_on_high` | `bool` | `False` | Treat as failure on high issues |
| `timeout_seconds` | `int` | `3600` | Execution timeout (seconds) |
| `sample_size` | `int` | `None` | Sampling size (None = all rows) |
| `tags` | `dict[str, str]` | `{}` | Tags (for routing, filtering) |
| `metadata` | `dict[str, Any]` | `{}` | Metadata |

## Checkpoint Execution

### Synchronous Execution

```python
# Single execution
result = checkpoint.run()

# Check result
print(result.status)           # CheckpointStatus.SUCCESS/FAILURE/ERROR/WARNING
print(result.run_id)           # Unique execution ID
print(result.duration_ms)      # Execution time (ms)
print(result.summary())        # Summary string

# Access validation results
validation = result.validation_result
print(validation.statistics.total_issues)
print(validation.statistics.pass_rate)

# Check action results
for action_result in result.action_results:
    print(f"{action_result.action_name}: {action_result.status}")
```

### CLI Execution

```bash
# Execute checkpoint from YAML configuration file
truthound checkpoint run daily_data_validation --config truthound.yaml

# Ad-hoc execution
truthound checkpoint run quick_check \
    --data data.csv \
    --validators null,duplicate

# Strict mode (exit code 1 on issues found)
truthound checkpoint run my_check --config truthound.yaml --strict

# JSON output
truthound checkpoint run my_check --format json --output result.json

# Include GitHub Actions summary
truthound checkpoint run my_check --github-summary
```

## CheckpointStatus

The status of an execution result.

```python
from truthound.checkpoint import CheckpointStatus

class CheckpointStatus(str, Enum):
    SUCCESS = "success"    # All validations passed
    FAILURE = "failure"    # Critical/High issues found (based on fail_on_* settings)
    ERROR = "error"        # Error occurred during execution
    WARNING = "warning"    # Issues found but within acceptable range
    RUNNING = "running"    # Currently executing
    PENDING = "pending"    # Waiting to execute
```

### Status Determination Logic

```python
# CheckpointResult status is determined by the following logic
def determine_status(validation_result, config):
    stats = validation_result.statistics

    # ERROR if execution error occurred
    if validation_result.error:
        return CheckpointStatus.ERROR

    # FAILURE if critical issues + fail_on_critical=True
    if config.fail_on_critical and stats.critical_issues > 0:
        return CheckpointStatus.FAILURE

    # FAILURE if high issues + fail_on_high=True
    if config.fail_on_high and stats.high_issues > 0:
        return CheckpointStatus.FAILURE

    # WARNING if issues exist
    if stats.total_issues > 0:
        return CheckpointStatus.WARNING

    return CheckpointStatus.SUCCESS
```

## CheckpointResult

A dataclass containing execution results.

```python
@dataclass
class CheckpointResult:
    run_id: str                              # Unique execution ID
    checkpoint_name: str                     # Checkpoint name
    run_time: datetime                       # Execution start time
    status: CheckpointStatus                 # Result status
    validation_result: ValidationResult      # Validation result object
    action_results: list[ActionResult]       # List of action execution results
    data_asset: str                          # Validated data asset name
    duration_ms: float                       # Total duration (milliseconds)
    error: str | None                        # Error message (on error)
    metadata: dict[str, Any]                 # User metadata
```

### Result Serialization

```python
# Convert to dictionary
data = result.to_dict()

# Save as JSON
import json
with open("result.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

# Restore from dictionary
restored = CheckpointResult.from_dict(data)
```

## Adding Actions

```python
from truthound.checkpoint.actions import (
    StoreValidationResult,
    SlackNotification,
    WebhookAction,
)

checkpoint = Checkpoint(
    name="with_actions",
    data_source="data.csv",
    validators=["null"],
    actions=[
        # Always store results
        StoreValidationResult(
            store_path="./results",
            partition_by="date",
            notify_on="always",
        ),
        # Slack notification on failure
        SlackNotification(
            webhook_url="https://hooks.slack.com/services/...",
            channel="#data-quality",
            notify_on="failure",
        ),
        # Webhook call
        WebhookAction(
            url="https://api.example.com/webhook",
            method="POST",
            notify_on="failure_or_error",
        ),
    ],
)
```

## Adding Triggers

Configure triggers for automatic execution.

```python
from truthound.checkpoint.triggers import ScheduleTrigger, CronTrigger

# Execute every hour
checkpoint = Checkpoint(
    name="hourly_check",
    data_source="data.csv",
    validators=["null"],
)
checkpoint.add_trigger(ScheduleTrigger(interval_hours=1))

# Use cron expression
checkpoint.add_trigger(CronTrigger(expression="0 9 * * 1"))  # Every Monday at 9 AM
```

## CheckpointRunner

Automatically executes multiple checkpoints.

```python
from truthound.checkpoint import CheckpointRunner

runner = CheckpointRunner(
    max_workers=4,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
    error_callback=lambda e: print(f"Error: {e}"),
)

# Add checkpoints
runner.add_checkpoint(checkpoint1)
runner.add_checkpoint(checkpoint2)

# Start background execution (trigger-based)
runner.start()

# Execute specific checkpoint once
result = runner.run_once("checkpoint1")

# Execute all checkpoints
results = runner.run_all()

# Iterate results
for result in runner.iter_results(timeout=1.0):
    print(result.summary())

# Shutdown
runner.stop()
```

## Registry

Register checkpoints to a global registry for access by name.

```python
from truthound.checkpoint import (
    CheckpointRegistry,
    register_checkpoint,
    get_checkpoint,
    list_checkpoints,
    load_checkpoints,
)

# Use global registry
register_checkpoint(checkpoint)

# Lookup by name
cp = get_checkpoint("my_check")
result = cp.run()

# List checkpoints
names = list_checkpoints()  # ['my_check', ...]

# Load from YAML
checkpoints = load_checkpoints("truthound.yaml")
for cp in checkpoints:
    register_checkpoint(cp)

# Custom registry
registry = CheckpointRegistry()
registry.register(checkpoint)

if "my_check" in registry:
    cp = registry.get("my_check")
```

## Next Steps

- [Actions Detail](./actions/index.md) - 14 action types explained
- [Triggers Detail](./triggers.md) - 4 trigger types
- [Routing](./routing.md) - Rule-based action routing
- [Async Execution](./async.md) - Asynchronous execution
