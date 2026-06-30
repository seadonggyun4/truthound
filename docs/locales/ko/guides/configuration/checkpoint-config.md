# 체크포인트 설정

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig

config = CheckpointConfig(
    name="daily_data_validation",
    data_source="users_table",
    validators=["not_null", "unique"],
    fail_on_critical=True,
)

checkpoint = Checkpoint(config)
result = checkpoint.run()
```

## 체크포인트Config

```python
from truthound.checkpoint import CheckpointConfig

config = CheckpointConfig(
    name="my_checkpoint",
    data_source="",                     # Data source identifier
    validators=None,                    # Validators to run
    validator_config={},                # Validator-specific config
    min_severity=None,                  # Minimum severity to report
    schema=None,                        # Schema path or object
    auto_schema=False,                  # Auto-generate schema
    run_name_template="%Y%m%d_%H%M%S",  # Run ID template
    tags={},                            # Custom tags
    metadata={},                        # Custom metadata
    fail_on_critical=True,              # Fail on critical issues
    fail_on_high=False,                 # Fail on high severity
    timeout_seconds=3600,               # Timeout (1 hour)
    sample_size=None,                   # Sample size (None = full)
)
```

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 name |
| 실무 운영 가이드에서 `fail_on_critical`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Fail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fail_on_high`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Fail을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `timeout_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `auto_schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Auto-generate 스키마 |

## Action 설정

### ActionConfig Base

```python
from truthound.checkpoint.actions.base import ActionConfig, NotifyCondition

config = ActionConfig(
    name=None,                          # Action name
    enabled=True,                       # Enable/disable
    notify_on=NotifyCondition.ALWAYS,   # When to notify
    timeout_seconds=30,                 # Action timeout
    retry_count=0,                      # Retry attempts
    retry_delay_seconds=1.0,            # Retry delay
    fail_checkpoint_on_error=False,     # Fail checkpoint on error
    metadata={},                        # Custom metadata
)
```

실무 운영 가이드에서 Notify, Conditions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| 실무 운영 가이드에서 `ALWAYS`, ALWAYS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Always을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SUCCESS`, SUCCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Only을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FAILURE`, FAILURE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Only on 실패 |
| 실무 운영 가이드에서 `ERROR`, ERROR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Only을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WARNING`, WARNING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Only을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FAILURE_OR_ERROR`, FAILURE_OR_ERROR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | On 실패 or error |
| 실무 운영 가이드에서 `NOT_SUCCESS`, NOT_SUCCESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Slack Notification

```python
from truthound.checkpoint.actions.slack_notify import (
    SlackNotification,
    SlackConfig,
)

config = SlackConfig(
    webhook_url="https://hooks.slack.com/...",
    channel="#data-quality",            # Override channel
    username="Truthound",
    icon_emoji=":mag:",
    include_details=True,               # Include issue details
    mention_on_failure=["@oncall"],     # Users to mention
    custom_message=None,                # Custom message template
    notify_on=NotifyCondition.FAILURE,
)

action = SlackNotification(config)
```

### Email Notification

```python
from truthound.checkpoint.actions.email_notify import (
    EmailNotification,
    EmailConfig,
)

config = EmailConfig(
    # SMTP settings
    smtp_host="smtp.example.com",
    smtp_port=587,
    smtp_user="user@example.com",
    smtp_password="secret",
    use_tls=True,
    use_ssl=False,

    # Email settings
    from_address="truthound@example.com",
    to_addresses=["team@example.com"],
    cc_addresses=[],
    subject_template="[Truthound] {status} - {checkpoint}",
    include_html=True,

    # Provider (smtp, sendgrid, ses)
    provider="smtp",
    api_key=None,                       # For SendGrid/SES

    notify_on=NotifyCondition.FAILURE,
)

action = EmailNotification(config)
```

### PagerDuty

```python
from truthound.checkpoint.actions.pagerduty import (
    PagerDutyAction,
    PagerDutyConfig,
)

config = PagerDutyConfig(
    routing_key="your-routing-key",
    severity="error",                   # critical, error, warning, info
    auto_severity=True,                 # Map from validation severity
    component="data-quality",
    group="truthound",
    class_type="validation",
    custom_details={},
    dedup_key_template="{checkpoint}_{data_asset}",
    resolve_on_success=True,            # Auto-resolve on success
    api_endpoint="https://events.pagerduty.com/v2/enqueue",
    notify_on=NotifyCondition.FAILURE_OR_ERROR,
)

action = PagerDutyAction(config)
```

### Webhook

```python
from truthound.checkpoint.actions.webhook import (
    WebhookAction,
    WebhookConfig,
)

config = WebhookConfig(
    url="https://api.example.com/webhook",
    method="POST",
    headers={"Content-Type": "application/json"},

    # Authentication
    auth_type="bearer",                 # none, basic, bearer, api_key
    auth_credentials={"token": "xxx"},

    payload_template=None,              # Custom payload template
    include_full_result=True,
    ssl_verify=True,
    success_codes=[200, 201, 202, 204],
    notify_on=NotifyCondition.ALWAYS,
)

action = WebhookAction(config)
```

### Store 결과

```python
from truthound.checkpoint.actions.store_result import (
    StoreResultAction,
    StoreResultConfig,
)

config = StoreResultConfig(
    store_path="./truthound_results",
    store_type="file",                  # file, s3, gcs, database
    format="json",                      # json, yaml, parquet
    partition_by="date",                # date, checkpoint, status
    retention_days=0,                   # 0 = unlimited
    include_validation_details=True,
    compress=False,
    notify_on=NotifyCondition.ALWAYS,
)

action = StoreResultAction(config)
```

## Routing 설정

### ActionRouter

```python
from truthound.checkpoint.routing.base import (
    ActionRouter,
    Route,
    RouteMode,
    RoutePriority,
)

router = ActionRouter(
    mode=RouteMode.ALL_MATCHES,         # Routing mode
    default_actions=[],                 # Fallback actions
)
```

실무 운영 가이드에서 Route, Modes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Mode을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `FIRST_MATCH`, FIRST_MATCH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Stop을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ALL_MATCHES`, ALL_MATCHES을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PRIORITY_GROUP`, PRIORITY_GROUP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Route, Priority을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Priority을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------|
| 실무 운영 가이드에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `HIGH`, HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NORMAL`, NORMAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `LOW`, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DEFAULT`, DEFAULT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Route 설정

```python
from truthound.checkpoint.routing.rules import (
    SeverityRule,
    IssueCountRule,
    StatusRule,
    PassRateRule,
    AllOf,
    AnyOf,
)

# Create routes
route = Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[pagerduty_action],
    priority=RoutePriority.CRITICAL,
    enabled=True,
    stop_on_match=False,
)

router.add_route(route)

# Complex rule with combinators
complex_rule = AllOf([
    SeverityRule(min_severity="high"),
    IssueCountRule(min_issues=10),
])

router.add_route(Route(
    name="high_volume_alerts",
    rule=complex_rule,
    actions=[slack_action, email_action],
))
```

### Available Rules

| 실무 운영 가이드에서 Rule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `SeverityRule`, SeverityRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `IssueCountRule`, IssueCountRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `StatusRule`, StatusRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TagRule`, TagRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PassRateRule`, PassRateRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TimeWindowRule`, TimeWindowRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DataAssetRule`, DataAssetRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Match by data 자산 |
| 실무 운영 가이드에서 `MetadataRule`, MetadataRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ErrorRule`, ErrorRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AlwaysRule`, AlwaysRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Always을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NeverRule`, NeverRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Never을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Combinators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Combinator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------|
| 실무 운영 가이드에서 `AllOf`, AllOf을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 AND을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AnyOf`, AnyOf을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NotRule`, NotRule을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Negate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### YAML 설정

```yaml
# routing.yaml
mode: all_matches

default_actions:
  - type: slack
    webhook_url: "${SLACK_WEBHOOK}"
    notify_on: failure

routes:
  - name: critical_alerts
    rule:
      type: severity
      min_severity: critical
    actions:
      - type: pagerduty
        routing_key: "${PAGERDUTY_KEY}"
    priority: critical
    enabled: true

  - name: high_volume
    rule:
      type: all_of
      rules:
        - type: severity
          min_severity: high
        - type: issue_count
          min_count: 10
    actions:
      - type: email
        to_addresses: ["team@example.com"]
```

```python
from truthound.checkpoint.routing.config import RouteConfigParser

parser = RouteConfigParser()
router = parser.parse_file("routing.yaml")
```

## Deduplication 설정

실무 운영 가이드에서 Prevent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.deduplication.protocols import (
    TimeWindow,
    WindowUnit,
    NotificationFingerprint,
)

# Time window configuration
window = TimeWindow(
    value=300,
    unit=WindowUnit.SECONDS,
)

# Convenience constructors
window = TimeWindow(minutes=5)
window = TimeWindow(hours=1)
window = TimeWindow(days=1)

# Generate fingerprint
fingerprint = NotificationFingerprint.generate(
    checkpoint_name="daily_data_validation",
    action_type="slack",
    severity="critical",
    data_asset="users_table",
)

# From checkpoint result
fingerprint = NotificationFingerprint.from_checkpoint_result(
    checkpoint_result,
    action_type="slack",
    include_issues=True,
    include_metadata=False,
)
```

### Window Strategies

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `SlidingWindowStrategy`, SlidingWindowStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Rolling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TumblingWindowStrategy`, TumblingWindowStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Fixed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SessionWindowStrategy`, SessionWindowStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Activity-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AdaptiveStrategy`, AdaptiveStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Adaptive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Deduplication 스토어

```python
from truthound.checkpoint.deduplication.stores import (
    InMemoryDeduplicationStore,
    RedisStreamsDeduplicationStore,
)

# In-memory (development)
store = InMemoryDeduplicationStore(max_size=10000)

# Redis Streams (production)
store = RedisStreamsDeduplicationStore(
    redis_url="redis://localhost:6379",
    stream_prefix="truthound:dedup:",
    max_stream_length=10000,
)
```

## Throttling 설정

실무 운영 가이드에서 Rate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.throttling.protocols import (
    ThrottlingConfig,
    RateLimit,
    RateLimitScope,
    TimeUnit,
)

config = ThrottlingConfig(
    per_minute_limit=10,
    per_hour_limit=100,
    per_day_limit=500,
    burst_multiplier=1.0,               # Allow burst
    scope=RateLimitScope.GLOBAL,        # Rate limit scope
    algorithm="token_bucket",           # Algorithm
    enabled=True,
    priority_bypass=False,              # Bypass for critical
    priority_threshold="critical",
    queue_on_throttle=False,            # Queue throttled
    max_queue_size=1000,
)

# Custom rate limits
custom_limits = {
    "slack": [
        RateLimit.per_minute(5),
        RateLimit.per_hour(50),
    ],
    "email": [
        RateLimit.per_hour(10),
        RateLimit.per_day(50),
    ],
}
```

실무 운영 가이드에서 Rate, Limit, Scopes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Scope을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `GLOBAL`, GLOBAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PER_ACTION`, PER_ACTION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Per을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PER_CHECKPOINT`, PER_CHECKPOINT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per 체크포인트 |
| 실무 운영 가이드에서 `PER_ACTION_CHECKPOINT`, PER_ACTION_CHECKPOINT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per action + 체크포인트 |
| 실무 운영 가이드에서 `PER_SEVERITY`, PER_SEVERITY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Per을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PER_DATA_ASSET`, PER_DATA_ASSET을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per data 자산 |
| 실무 운영 가이드에서 `CUSTOM`, CUSTOM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Throttlers

```python
from truthound.checkpoint.throttling.throttlers import (
    TokenBucketThrottler,
    FixedWindowThrottler,
    SlidingWindowThrottler,
    CompositeThrottler,
    ThrottlerBuilder,
)

# Token bucket (default)
throttler = TokenBucketThrottler(config)

# Fixed window
throttler = FixedWindowThrottler(config)

# Sliding window
throttler = SlidingWindowThrottler(config)

# Builder pattern
throttler = (
    ThrottlerBuilder()
    .with_limit(RateLimit.per_minute(10))
    .with_limit(RateLimit.per_hour(100))
    .with_scope(RateLimitScope.PER_ACTION)
    .with_burst(1.5)
    .build()
)
```

## Escalation 설정

실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.escalation.protocols import (
    EscalationTarget,
    EscalationTrigger,
    TargetType,
)

# Define escalation targets
target = EscalationTarget(
    type=TargetType.USER,
    identifier="@oncall",
    name="On-Call Engineer",
    priority=1,
)

# Factory methods
target = EscalationTarget.user("@john", name="John Doe")
target = EscalationTarget.team("data-team", name="Data Team")
```

실무 운영 가이드에서 Target, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `USER`, USER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Individual을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TEAM`, TEAM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Team/group을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CHANNEL`, CHANNEL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Chat을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SCHEDULE`, SCHEDULE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | On-call 스케줄 |
| 실무 운영 가이드에서 `WEBHOOK`, WEBHOOK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Webhook을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `EMAIL`, EMAIL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Email을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PHONE`, PHONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Phone을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CUSTOM`, CUSTOM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 Escalation, Triggers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Trigger을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|
| 실무 운영 가이드에서 `UNACKNOWLEDGED`, UNACKNOWLEDGED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Not을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `UNRESOLVED`, UNRESOLVED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Not을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SEVERITY_UPGRADE`, SEVERITY_UPGRADE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `REPEATED_FAILURE`, REPEATED_FAILURE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Multiple 실패 |
| 실무 운영 가이드에서 `THRESHOLD_BREACH`, THRESHOLD_BREACH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Threshold을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MANUAL`, MANUAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Manual을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SCHEDULED`, SCHEDULED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Scheduled을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Escalation 스토어

```python
from truthound.checkpoint.escalation.stores import (
    InMemoryEscalationStore,
    RedisEscalationStore,
    SQLiteEscalationStore,
)

# In-memory
store = InMemoryEscalationStore()

# Redis
store = RedisEscalationStore(
    redis_url="redis://localhost:6379",
    prefix="truthound:escalation:",
)

# SQLite
store = SQLiteEscalationStore(
    db_path="./escalation.db",
)
```

## Complete Example

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.checkpoint.actions import SlackNotification, EmailNotification
from truthound.checkpoint.routing import ActionRouter, Route, RouteMode
from truthound.checkpoint.routing.rules import SeverityRule, AllOf, IssueCountRule

# Configure checkpoint
config = CheckpointConfig(
    name="daily_user_validation",
    data_source="users_table",
    validators=["not_null", "unique", "email_format"],
    fail_on_critical=True,
    timeout_seconds=1800,
    tags={"environment": "production"},
)

# Configure actions
slack = SlackNotification(SlackConfig(
    webhook_url="${SLACK_WEBHOOK}",
    channel="#data-quality",
    notify_on="failure",
))

email = EmailNotification(EmailConfig(
    smtp_host="smtp.example.com",
    to_addresses=["team@example.com"],
    notify_on="failure",
))

# Configure router
router = ActionRouter(mode=RouteMode.ALL_MATCHES)

router.add_route(Route(
    name="critical_alerts",
    rule=SeverityRule(min_severity="critical"),
    actions=[slack, email],
    priority="critical",
))

router.add_route(Route(
    name="high_volume",
    rule=AllOf([
        SeverityRule(min_severity="high"),
        IssueCountRule(min_issues=10),
    ]),
    actions=[slack],
    priority="high",
))

# Create and run checkpoint
checkpoint = Checkpoint(config)
checkpoint.set_router(router)
result = checkpoint.run()

print(f"Status: {result.status}")
if result.validation_view is not None:
    print(f"Issues: {result.validation_view.statistics.total_issues}")
```

## 환경 변수

```bash
# Slack
export SLACK_WEBHOOK=https://hooks.slack.com/...

# PagerDuty
export PAGERDUTY_KEY=your-routing-key

# Email
export SMTP_HOST=smtp.example.com
export SMTP_USER=user@example.com
export SMTP_PASSWORD=secret

# Redis (for deduplication/throttling)
export REDIS_URL=redis://localhost:6379
```
