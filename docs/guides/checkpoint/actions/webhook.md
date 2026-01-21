# Webhook Actions

Actions for integration with external systems via HTTP webhooks.

## WebhookAction

A general-purpose webhook action that can send requests to any HTTP endpoint.

### Configuration (WebhookConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `url` | `str` | `""` | Webhook URL (required) |
| `method` | `str` | `"POST"` | HTTP method: `GET`, `POST`, `PUT`, `PATCH`, `DELETE` |
| `headers` | `dict[str, str]` | `{}` | Additional HTTP headers |
| `auth_type` | `str` | `"none"` | Authentication type: `none`, `basic`, `bearer`, `api_key` |
| `auth_credentials` | `dict[str, str]` | `{}` | Authentication credentials |
| `payload_template` | `dict \| None` | `None` | Custom payload template |
| `include_full_result` | `bool` | `True` | Include full result in payload |
| `ssl_verify` | `bool` | `True` | SSL certificate verification |
| `success_codes` | `list[int]` | `[200, 201, 202, 204]` | HTTP status codes considered successful |
| `notify_on` | `str` | `"always"` | Execution condition |

### Basic Usage

```python
from truthound.checkpoint.actions import WebhookAction

# Basic POST request
action = WebhookAction(
    url="https://api.example.com/data-quality/events",
    notify_on="failure",
)

# PUT request
action = WebhookAction(
    url="https://api.example.com/status",
    method="PUT",
    notify_on="always",
)
```

### Authentication Configuration

#### Bearer Token Authentication

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="bearer",
    auth_credentials={
        "token": "${API_TOKEN}",  # Environment variable reference
    },
)
```

#### Basic Authentication

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="basic",
    auth_credentials={
        "username": "user",
        "password": "${API_PASSWORD}",
    },
)
```

#### API Key Authentication

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="api_key",
    auth_credentials={
        "header": "X-API-Key",  # Header name (default: "X-API-Key")
        "key": "${API_KEY}",
    },
)
```

### Custom Headers

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    headers={
        "X-Custom-Header": "custom-value",
        "X-Request-ID": "${REQUEST_ID}",
        "Accept": "application/json",
    },
)
```

### Custom Payload

Use `payload_template` to customize the payload. Placeholders are supported.

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    payload_template={
        "event_type": "data_quality_check",
        "checkpoint": "${checkpoint}",
        "status": "${status}",
        "run_id": "${run_id}",
        "timestamp": "${run_time}",
        "metrics": {
            "total_issues": "${total_issues}",
            "critical": "${critical_issues}",
            "high": "${high_issues}",
            "pass_rate": "${pass_rate}",
        },
        "custom_field": "custom_value",
    },
)
```

#### Supported Placeholders

| Placeholder | Description |
|-------------|-------------|
| `${checkpoint}` | Checkpoint name |
| `${run_id}` | Execution ID |
| `${status}` | Result status |
| `${run_time}` | Execution time (ISO 8601) |
| `${data_asset}` | Data asset name |
| `${total_issues}` | Total issue count |
| `${critical_issues}` | Critical issue count |
| `${high_issues}` | High issue count |
| `${medium_issues}` | Medium issue count |
| `${low_issues}` | Low issue count |
| `${pass_rate}` | Pass rate |

### Default Payload

Default payload when `payload_template` is not specified:

```json
{
  "event": "validation_completed",
  "checkpoint": "daily_data_validation",
  "run_id": "20240115_120000",
  "status": "failure",
  "run_time": "2024-01-15T12:00:00",
  "data_asset": "users.csv",
  "summary": {
    "total_issues": 150,
    "critical_issues": 5,
    "high_issues": 25,
    "medium_issues": 70,
    "low_issues": 50,
    "pass_rate": 0.85
  },
  "full_result": { ... }  // When include_full_result=True
}
```

### Disable SSL Verification

For self-signed certificates on internal networks:

```python
action = WebhookAction(
    url="https://internal.example.com/webhook",
    ssl_verify=False,  # Warning: Not recommended for security reasons
)
```

### Custom Success Codes

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    success_codes=[200, 201, 202, 204, 302],  # Treat 302 redirect as success
)
```

### Retry Configuration

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    timeout_seconds=30,    # Request timeout
    retry_count=3,         # Maximum 3 retries on failure
    retry_delay_seconds=2, # 2-second interval between retries
)
```

---

## GitHubAction

Action for integration with GitHub Actions. Configures Job Summary, Annotations, and Outputs.

### Configuration

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `token` | `str` | `""` | GitHub Token (uses environment variable if not provided) |
| `repo` | `str` | `""` | Repository (owner/repo) |
| `check_name` | `str` | `"Truthound"` | Check Run name |
| `step_summary` | `bool` | `True` | Write Job Summary |
| `set_output` | `bool` | `True` | Set workflow outputs |
| `annotations` | `bool` | `True` | Output error/warning annotations |
| `notify_on` | `str` | `"always"` | Execution condition |

### Usage Example

```python
from truthound.checkpoint.actions import GitHubAction

action = GitHubAction(
    token="${GITHUB_TOKEN}",
    repo="owner/repo",
    step_summary=True,
    set_output=True,
    annotations=True,
)
```

### Usage in GitHub Actions Workflow

```yaml
- name: Run Data Quality Check
  run: truthound checkpoint run my_check --config config.yaml
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

- name: Use Outputs
  run: |
    echo "Status: ${{ steps.dq-check.outputs.status }}"
    echo "Issues: ${{ steps.dq-check.outputs.total_issues }}"
```

---

## YAML Configuration Example

```yaml
actions:
  # Basic webhook
  - type: webhook
    url: https://api.example.com/data-quality/events
    method: POST
    notify_on: failure

  # Authentication configuration
  - type: webhook
    url: https://api.example.com/webhook
    method: POST
    auth_type: bearer
    auth_credentials:
      token: ${API_TOKEN}
    headers:
      X-Custom-Header: custom-value
    notify_on: always

  # Custom payload
  - type: webhook
    url: https://api.example.com/webhook
    payload_template:
      event: data_quality
      checkpoint: "${checkpoint}"
      status: "${status}"
      issues: "${total_issues}"
    include_full_result: false
    notify_on: failure_or_error

  # GitHub Actions integration
  - type: github
    step_summary: true
    set_output: true
    annotations: true
    notify_on: always
```
