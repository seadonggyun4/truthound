# Custom Actions

Actions for executing user-defined logic.

## CustomAction

Executes Python callback functions or shell commands.

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `callback` | `Callable \| None` | `None` | Python callback function |
| `shell_command` | `str \| None` | `None` | Shell command |
| `environment` | `dict[str, str]` | `{}` | Environment variables |
| `pass_result_as_json` | `bool` | `True` | Pass result as JSON (for shell commands) |
| `working_directory` | `str \| None` | `None` | Working directory |
| `notify_on` | `str` | `"always"` | Execution condition |

### Using Python Callbacks

```python
from truthound.checkpoint.actions import CustomAction

def my_callback(checkpoint_result):
    """Custom logic for processing validation results."""
    status = checkpoint_result.status.value
    stats = checkpoint_result.validation_result.statistics

    print(f"Checkpoint {checkpoint_result.checkpoint_name}: {status}")
    print(f"Total issues: {stats.total_issues}")

    if status == "failure":
        # Custom notification logic
        send_custom_alert(checkpoint_result)

    # Save additional data
    save_to_database(checkpoint_result)

    # Return value is included in ActionResult.details
    return {"processed": True, "custom_metric": 42}


action = CustomAction(
    callback=my_callback,
    notify_on="always",
)
```

### Async Callbacks

```python
import asyncio

async def async_callback(checkpoint_result):
    """Asynchronous custom logic."""
    await asyncio.sleep(1)  # Async operation
    await send_notification_async(checkpoint_result)
    return {"async_result": True}


action = CustomAction(callback=async_callback)
```

### Using Shell Commands

```python
# Simple shell command
action = CustomAction(
    shell_command="./scripts/notify.sh",
    notify_on="failure",
)

# Pass environment variables
action = CustomAction(
    shell_command="./scripts/process_result.py",
    environment={
        "API_KEY": "${SECRET_KEY}",
        "ENVIRONMENT": "production",
    },
    pass_result_as_json=True,  # Pass result to stdin
    working_directory="./scripts",
)
```

### Shell Script Examples

When `pass_result_as_json=True`, the result is passed to stdin:

```bash
#!/bin/bash
# scripts/process_result.sh

# Read JSON from stdin
result=$(cat)

# Parse with jq
status=$(echo $result | jq -r '.status')
issues=$(echo $result | jq -r '.validation_result.statistics.total_issues')
checkpoint=$(echo $result | jq -r '.checkpoint_name')

echo "Checkpoint: $checkpoint"
echo "Status: $status"
echo "Issues: $issues"

# Conditional processing
if [ "$status" = "failure" ]; then
    curl -X POST "https://api.example.com/alert" \
        -H "Content-Type: application/json" \
        -d "{\"checkpoint\": \"$checkpoint\", \"issues\": $issues}"
fi
```

Python script example:

```python
#!/usr/bin/env python3
# scripts/process_result.py

import json
import sys

# Read result from stdin
result = json.load(sys.stdin)

checkpoint = result["checkpoint_name"]
status = result["status"]
stats = result["validation_result"]["statistics"]

print(f"Processing {checkpoint}: {status}")
print(f"Issues: {stats['total_issues']}")

# Custom logic...
```

### Conditional Execution

```python
def conditional_callback(checkpoint_result):
    """Logic that executes only under specific conditions."""
    stats = checkpoint_result.validation_result.statistics

    # Page only when 10+ critical issues
    if stats.critical_issues >= 10:
        page_on_call_engineer(checkpoint_result)
        return {"paged": True}

    return {"paged": False}


action = CustomAction(
    callback=conditional_callback,
    notify_on="failure",  # Callback invoked only on failure
)
```

### Error Handling

When an exception occurs in the callback, the ActionResult status becomes ERROR:

```python
def risky_callback(checkpoint_result):
    try:
        # Risky operation
        result = do_something_risky()
        return {"success": True, "result": result}
    except Exception as e:
        # Re-raising the exception records it in ActionResult.error
        raise RuntimeError(f"Failed to process: {e}")


# Or return failure explicitly
def safe_callback(checkpoint_result):
    try:
        result = do_something_risky()
        return {"success": True}
    except Exception as e:
        # Catch exception and return failure info
        return {"success": False, "error": str(e)}
```

### Combining with Other Actions

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.actions import (
    StoreValidationResult,
    SlackNotification,
    CustomAction,
)

def post_process(result):
    """Execute after all other actions complete."""
    # Post-process results
    aggregate_metrics(result)
    update_dashboard(result)
    return {"post_processed": True}


checkpoint = Checkpoint(
    name="my_check",
    data_source="data.csv",
    validators=["null"],
    actions=[
        # Executed in order
        StoreValidationResult(store_path="./results"),      # 1. Store
        SlackNotification(webhook_url="...", notify_on="failure"),  # 2. Notify
        CustomAction(callback=post_process),                # 3. Post-process
    ],
)
```

---

## YAML Configuration Examples

```yaml
actions:
  # Shell command
  - type: custom
    shell_command: ./scripts/notify.sh
    environment:
      API_KEY: ${API_KEY}
    pass_result_as_json: true
    notify_on: failure

  # Python script
  - type: custom
    shell_command: python ./scripts/process.py
    working_directory: ./scripts
    pass_result_as_json: true
    notify_on: always
```

Note: Python callbacks cannot be specified directly in YAML. Use the Python API for complex logic.
