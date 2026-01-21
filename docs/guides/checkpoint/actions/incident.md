# Incident Actions

Actions for integration with incident management systems. Supports PagerDuty and OpsGenie.

## PagerDutyAction

Creates and manages incidents through the PagerDuty Events API v2.

### Configuration (PagerDutyConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `routing_key` | `str` | `""` | Events API v2 routing key (Integration Key) |
| `severity` | `str` | `"error"` | Default severity: `critical`, `error`, `warning`, `info` |
| `auto_severity` | `bool` | `True` | Automatic severity mapping based on validation results |
| `component` | `str` | `"data-quality"` | Component name |
| `group` | `str` | `"truthound"` | Logical group |
| `class_type` | `str` | `"validation"` | Incident class/type |
| `custom_details` | `dict` | `{}` | Additional custom details |
| `dedup_key_template` | `str` | `"{checkpoint}_{data_asset}"` | Deduplication key template |
| `resolve_on_success` | `bool` | `True` | Automatically resolve incident on success |
| `api_endpoint` | `str` | `"https://events.pagerduty.com/v2/enqueue"` | API endpoint |
| `notify_on` | `str` | `"failure_or_error"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import PagerDutyAction

# Basic usage
action = PagerDutyAction(
    routing_key="${PAGERDUTY_ROUTING_KEY}",
    auto_severity=True,
    resolve_on_success=True,
)

# Detailed configuration
action = PagerDutyAction(
    routing_key="${PAGERDUTY_ROUTING_KEY}",
    severity="critical",  # Used when auto_severity=False
    auto_severity=False,
    component="production-etl",
    group="data-platform",
    class_type="data-quality-validation",
    custom_details={
        "team": "data-engineering",
        "runbook": "https://wiki.example.com/dq-runbook",
    },
    resolve_on_success=True,
    notify_on="failure_or_error",
)

# Custom deduplication key
action = PagerDutyAction(
    routing_key="...",
    dedup_key_template="{checkpoint}_{data_asset}_{run_id}",  # Separate incident per execution
)
```

### auto_severity Mapping

When `auto_severity=True`, PagerDuty severity mapping based on validation results:

| Condition | PagerDuty Severity |
|-----------|---------------------|
| `critical_issues > 0` | `critical` |
| `high_issues > 0` | `error` |
| `medium_issues > 0` | `warning` |
| Otherwise | `info` |

### Incident Lifecycle

1. **Trigger**: Creates incident on validation failure
2. **Dedup**: Prevents duplicate incidents with the same `dedup_key`
3. **Resolve**: Automatically resolves when `resolve_on_success=True` and validation succeeds

```python
# Example flow
# 1. First run - failure → Create incident
# 2. Second run - failure → Update existing incident (same dedup_key)
# 3. Third run - success → Resolve incident
```

### Payload Example

```json
{
  "routing_key": "xxx",
  "event_action": "trigger",
  "dedup_key": "daily_data_validation_users.csv",
  "payload": {
    "summary": "Data quality failure for 'daily_data_validation' on users.csv - 150 issues found (5 critical)",
    "severity": "critical",
    "source": "users.csv",
    "component": "data-quality",
    "group": "truthound",
    "class": "validation",
    "timestamp": "2024-01-15T12:00:00",
    "custom_details": {
      "checkpoint": "daily_data_validation",
      "run_id": "20240115_120000",
      "status": "failure",
      "data_asset": "users.csv",
      "statistics": {
        "total_issues": 150,
        "critical": 5,
        "high": 25,
        "medium": 70,
        "low": 50,
        "pass_rate": "85.0%"
      }
    }
  }
}
```

---

## OpsGenieAction

Creates and manages alerts through the OpsGenie Alert API.

### Configuration (OpsGenieConfig)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `api_key` | `str` | `""` | OpsGenie API key |
| `region` | `str` | `"us"` | Region: `us`, `eu` |
| `priority` | `AlertPriority` | `P3` | Default priority: `P1`-`P5` |
| `auto_priority` | `bool` | `True` | Automatic priority mapping based on validation results |
| `responders` | `list[Responder]` | `[]` | List of responders |
| `visible_to` | `list[Responder]` | `[]` | Alert visibility targets |
| `tags` | `list[str]` | `[]` | List of tags |
| `actions` | `list[str]` | `[]` | List of action buttons |
| `alias_template` | `str` | `"{checkpoint}_{data_asset}"` | Alert alias template (for deduplication) |
| `auto_close_on_success` | `bool` | `True` | Automatically close on success |
| `notify_on` | `str` | `"failure"` | Execution condition |

### Usage Examples

```python
from truthound.checkpoint.actions import OpsGenieAction
from truthound.checkpoint.actions.opsgenie import (
    AlertPriority,
    Responder,
    ResponderType,
)

# Basic usage
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    auto_priority=True,
)

# Responder configuration
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    responders=[
        Responder(type=ResponderType.TEAM, name="data-platform"),
        Responder(type=ResponderType.USER, username="oncall@example.com"),
        Responder(type=ResponderType.ESCALATION, name="data-quality-escalation"),
    ],
    visible_to=[
        Responder(type=ResponderType.TEAM, name="engineering"),
    ],
    priority=AlertPriority.P1,
    auto_priority=False,
    tags=["data-quality", "production", "automated"],
    auto_close_on_success=True,
)

# EU region
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    region="eu",  # Uses api.eu.opsgenie.com
)
```

### auto_priority Mapping

When `auto_priority=True`, OpsGenie priority mapping based on validation results:

| Condition | OpsGenie Priority |
|-----------|-------------------|
| `critical_issues > 0` | `P1` (Critical) |
| `high_issues > 0` | `P2` (High) |
| `medium_issues > 0` | `P3` (Moderate) |
| `low_issues > 0` | `P4` (Low) |
| Otherwise | `P5` (Informational) |

### Responder Types

```python
class ResponderType(str, Enum):
    TEAM = "team"              # Team
    USER = "user"              # User (username)
    ESCALATION = "escalation"  # Escalation policy
    SCHEDULE = "schedule"      # Schedule
```

### Factory Functions

```python
from truthound.checkpoint.actions.opsgenie import (
    create_opsgenie_action,
    create_critical_alert,
    create_team_alert,
)

# Create critical alert
action = create_critical_alert(
    api_key="${OPSGENIE_API_KEY}",
    team="data-platform",
    escalation_policy="data-quality-escalation",
)

# Create team alert
action = create_team_alert(
    api_key="${OPSGENIE_API_KEY}",
    team="data-platform",
)
```

---

## YAML Configuration Example

```yaml
actions:
  # PagerDuty
  - type: pagerduty
    routing_key: ${PAGERDUTY_ROUTING_KEY}
    auto_severity: true
    resolve_on_success: true
    component: production-etl
    group: data-platform
    notify_on: failure_or_error

  # OpsGenie
  - type: opsgenie
    api_key: ${OPSGENIE_API_KEY}
    region: us
    auto_priority: true
    responders:
      - type: team
        name: data-platform
      - type: user
        username: oncall@example.com
    tags:
      - data-quality
      - production
    auto_close_on_success: true
    notify_on: failure
```

## Comparison: PagerDuty vs OpsGenie

| Feature | PagerDuty | OpsGenie |
|---------|-----------|----------|
| **Incident Creation** | Events API v2 | Alert API |
| **Deduplication** | `dedup_key` | `alias` |
| **Auto Resolution** | `event_action: resolve` | Close Alert API |
| **Priority** | severity (4 levels) | priority (P1-P5) |
| **Responders** | Escalation policy | Team, User, Schedule, Policy |
| **Region** | Single | US, EU |
