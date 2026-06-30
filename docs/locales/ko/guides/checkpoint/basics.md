# 체크포인트 Basics

실무 운영 가이드에서 Checkpoint을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Creating a 체크포인트

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

### YAML 설정

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

## 체크포인트Config Properties

| 실무 운영 가이드에서 Property을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|------|---------|-------------|
| 실무 운영 가이드에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"default_checkpoint"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Unique 체크포인트 name |
| 실무 운영 가이드에서 `data_source`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Data 소스 path or object |
| 실무 운영 가이드에서 `validators`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증기]` | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | List of 검증기 to execute |
| 실무 운영 가이드에서 `validator_config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per-검증기 설정 |
| 실무 운영 가이드에서 `min_severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `critical`, `high`, `medium`, `low`, Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 파일 path |
| 실무 운영 가이드에서 `auto_schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Enable automatic 스키마 inference |
| 실무 운영 가이드에서 `run_name_template`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"%Y%m%d_%H%M%S"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fail_on_critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Treat as 실패 on critical issues |
| 실무 운영 가이드에서 `fail_on_high`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Treat as 실패 on high issues |
| 실무 운영 가이드에서 `timeout_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `3600`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sampling, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `tags`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict[str, str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tags을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `metadata`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict[str, Any]`, Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Metadata을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 체크포인트 Execution

### Synchronous Execution

```python
# Single execution
result = checkpoint.run()

# Check result
print(result.status)           # CheckpointStatus.SUCCESS/FAILURE/ERROR/WARNING
print(result.run_id)           # Unique execution ID
print(result.duration_ms)      # Execution time (ms)
print(result.summary())        # Summary string

# Access canonical validation results
validation_run = result.validation_run
if validation_run is not None:
    print(len(validation_run.checks))
    print(len(validation_run.issues))

# Access compatibility statistics via validation_view
if result.validation_view is not None:
    print(result.validation_view.statistics.total_issues)
    print(result.validation_view.statistics.pass_rate)

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

## 체크포인트Status

The status of an execution 결과.

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
def determine_status(result, config):
    if result.error:
        return CheckpointStatus.ERROR

    validation = result.validation_view
    if validation is None:
        return CheckpointStatus.ERROR

    stats = validation.statistics

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

## 체크포인트Result

A dataclass containing execution 결과.

```python
@dataclass
class CheckpointResult:
    run_id: str                              # Unique execution ID
    checkpoint_name: str                     # Checkpoint name
    run_time: datetime                       # Execution start time
    status: CheckpointStatus                 # Result status
    validation_run: ValidationRunResult | None  # Canonical result from th.check()
    action_results: list[ActionResult]       # List of action execution results
    data_asset: str                          # Validated data asset name
    duration_ms: float                       # Total duration (milliseconds)
    error: str | None                        # Error message (on error)
    metadata: dict[str, Any]                 # User metadata

    @property
    def validation_view(self) -> CheckpointValidationView | None:
        """Compatibility view derived from validation_run."""
```

### 결과 Serialization

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

실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## 체크포인트Runner

Automatically executes multiple 체크포인트.

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

실무 운영 가이드에서 Register을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## 다음 단계

- 실무 운영 가이드에서 Actions, Detail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Triggers, Detail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Routing, Rule-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Async, Execution, Asynchronous을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
