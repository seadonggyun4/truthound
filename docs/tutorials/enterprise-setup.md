# Enterprise Setup Tutorial

Learn how to set up Truthound for production environments with CI/CD integration, checkpoints, and notifications.

## Overview

This tutorial covers:

- CI/CD pipeline integration
- Checkpoint-based validation workflows
- Notification setup (Slack, Email, PagerDuty, etc.)
- Production-grade configuration

## Prerequisites

- Truthound installed (`pip install truthound[all]`)
- Access to your CI/CD platform (GitHub Actions, GitLab CI, etc.)
- Optional: Slack webhook URL, email SMTP, or other notification credentials

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/data-quality.yml`:

```yaml
name: Data Quality Check

on:
  push:
    paths:
      - 'data/**'
  pull_request:
    paths:
      - 'data/**'
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Truthound
        run: pip install truthound[all]

      - name: Run validation
        run: truthound check data/input.csv --strict --min-severity medium

      - name: Check for drift
        run: |
          truthound compare data/baseline.csv data/current.csv \
            --format json > drift.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: drift.json
```

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
data-quality:
  image: python:3.11
  stage: test
  script:
    - pip install truthound[all]
    - truthound check data/input.csv --strict --min-severity medium
    - truthound compare data/baseline.csv data/current.csv --format json > drift.json
  artifacts:
    reports:
      junit: validation-report.xml
    paths:
      - drift.json
  rules:
    - changes:
        - data/**/*
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Validate Data') {
            steps {
                sh 'pip install truthound[all]'
                sh 'truthound check data/input.csv --strict --min-severity high'
            }
        }
        stage('Check Drift') {
            steps {
                sh 'truthound compare data/baseline.csv data/current.csv --format json > drift.json'
                archiveArtifacts artifacts: 'drift.json'
            }
        }
    }
    post {
        failure {
            emailext subject: 'Data Quality Check Failed',
                     body: 'Check ${BUILD_URL} for details',
                     to: 'data-team@company.com'
        }
    }
}
```

### Python Script for CI

```python
#!/usr/bin/env python3
"""Data quality check for CI/CD pipelines."""
import sys
import truthound as th
from truthound.types import Severity

def main():
    # Run validation
    report = th.check("data/input.csv", min_severity="medium")

    # Check for critical issues
    if report.has_critical:
        critical_issues = [i for i in report.issues if i.severity == Severity.CRITICAL]
        print(f"CRITICAL: Found {len(critical_issues)} critical issues!")
        for issue in critical_issues:
            print(f"  - {issue.column}: {issue.issue_type}")
        sys.exit(1)

    # Check for drift
    drift = th.compare("data/baseline.csv", "data/current.csv")

    if drift.has_high_drift:
        print("WARNING: Significant drift detected!")
        for col_drift in drift.columns:
            if col_drift.result.drifted:
                print(f"  - {col_drift.column}: statistic={col_drift.result.statistic:.4f}")
        sys.exit(1)

    print(f"SUCCESS: All checks passed ({len(report.issues)} minor issues)")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

## Checkpoint System

Checkpoints provide a structured way to define, run, and track validation workflows.

### Basic Checkpoint

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig, CheckpointStatus

# Configure checkpoint
config = CheckpointConfig(
    name="daily_sales_validation",
    data_source="data/sales.parquet",
    validators=["null", "duplicate", "range", "outlier"],
    min_severity="medium",
    fail_on_critical=True,
    fail_on_high=False,
    timeout_seconds=300,
)

# Create and run checkpoint
checkpoint = Checkpoint(config=config)
result = checkpoint.run()

# Check result
if result.status == CheckpointStatus.SUCCESS:
    print("Validation passed!")
else:
    print(f"Validation failed: {result.error}")
    for action_result in result.action_results:
        print(f"  Action: {action_result.action_name} - {action_result.status}")
```

### Checkpoint with Actions

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.checkpoint.actions import (
    SlackNotification,
    EmailNotification,
    WebhookAction,
    NotifyCondition,
)

# Configure notifications
slack_action = SlackNotification(
    webhook_url="https://hooks.slack.com/services/xxx/yyy/zzz",
    channel="#data-alerts",
    notify_on=NotifyCondition.FAILURE,  # Only on failures
)

email_action = EmailNotification(
    smtp_host="smtp.company.com",
    smtp_port=587,
    from_addr="data-quality@company.com",
    to_addrs=["team@company.com"],
    notify_on=NotifyCondition.ALWAYS,  # Always notify
)

webhook_action = WebhookAction(
    url="https://api.company.com/webhooks/dq",
    method="POST",
    notify_on=NotifyCondition.ALWAYS,
)

# Create checkpoint with actions
config = CheckpointConfig(
    name="production_validation",
    data_source="data/production.parquet",
    validators=["null", "schema", "freshness"],
)

checkpoint = Checkpoint(config=config)
checkpoint.add_action(slack_action)
checkpoint.add_action(email_action)
checkpoint.add_action(webhook_action)

# Run - actions execute automatically based on result
result = checkpoint.run()
```

### Checkpoint YAML Configuration

Create `checkpoints/sales_checkpoint.yaml`:

```yaml
name: daily_sales_validation
data_source: data/sales.parquet

validators:
  - null
  - duplicate
  - range
  - outlier

validator_config:
  range:
    columns:
      amount:
        min: 0
        max: 1000000
  outlier:
    method: iqr
    threshold: 3.0

min_severity: medium
fail_on_critical: true
fail_on_high: false
timeout_seconds: 300

tags:
  environment: production
  team: data-platform

metadata:
  owner: data-team@company.com
  schedule: daily
```

Run with CLI:

```bash
truthound checkpoint run checkpoints/sales_checkpoint.yaml
```

### Checkpoint Registry

Register and manage multiple checkpoints:

```python
from truthound.checkpoint import CheckpointRegistry

# Create registry
registry = CheckpointRegistry()

# Register checkpoints from directory
registry.register_from_directory("checkpoints/")

# List all checkpoints
for name in registry.list_checkpoints():
    print(f"  - {name}")

# Run specific checkpoint
result = registry.run("daily_sales_validation")

# Run all checkpoints
results = registry.run_all()
for name, result in results.items():
    print(f"{name}: {result.status}")
```

## Notification Setup

### Slack Integration

```python
from truthound.checkpoint.actions import SlackNotification, NotifyCondition

slack = SlackNotification(
    webhook_url="https://hooks.slack.com/services/T00/B00/xxx",
    channel="#data-quality-alerts",
    username="Truthound Bot",
    icon_emoji=":mag:",
    notify_on=NotifyCondition.FAILURE,
)
```

### Email Integration

```python
from truthound.checkpoint.actions import EmailNotification, NotifyCondition

email = EmailNotification(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_user="alerts@company.com",
    smtp_password="app-specific-password",  # Use env var in production
    from_addr="Data Quality <alerts@company.com>",
    to_addrs=["team@company.com", "manager@company.com"],
    notify_on=NotifyCondition.FAILURE,
)
```

### PagerDuty Integration

```python
from truthound.checkpoint.actions import PagerDutyAction, NotifyCondition

pagerduty = PagerDutyAction(
    routing_key="your-pagerduty-routing-key",
    notify_on=NotifyCondition.FAILURE,
)
```

### Discord Integration

```python
from truthound.checkpoint.actions import DiscordNotification, NotifyCondition

discord = DiscordNotification(
    webhook_url="https://discord.com/api/webhooks/xxx/yyy",
    username="Truthound",
    notify_on=NotifyCondition.ALWAYS,
)
```

### Telegram Integration

```python
from truthound.checkpoint.actions import TelegramNotification, NotifyCondition

telegram = TelegramNotification(
    bot_token="123456:ABC-DEF...",
    chat_id="-1001234567890",
    notify_on=NotifyCondition.FAILURE,
)
```

### Custom Webhook

```python
from truthound.checkpoint.actions import WebhookAction

webhook = WebhookAction(
    url="https://api.company.com/data-quality/events",
    method="POST",
    headers={
        "Authorization": "Bearer ${DQ_API_TOKEN}",
        "Content-Type": "application/json",
    },
    payload_template={
        "checkpoint": "{checkpoint_name}",
        "status": "{status}",
        "issues": "{issue_count}",
        "timestamp": "{timestamp}",
    },
)
```

## Advanced Notification Features

### Rule-Based Routing

Route notifications based on validation results:

```python
from truthound.checkpoint.routing import (
    ActionRouter,
    Route,
    SeverityRule,
    IssueCountRule,
    TimeWindowRule,
    AllOf,
    AnyOf,
)

# Create routing rules
critical_rule = SeverityRule(min_severity="critical")
many_issues_rule = IssueCountRule(min_issues=100)
business_hours = TimeWindowRule(start_hour=9, end_hour=17, weekdays_only=True)

# Combine rules
escalate_rule = AnyOf([
    critical_rule,
    AllOf([many_issues_rule, business_hours]),
])

# Create router with routes
router = ActionRouter()
router.add_route(Route(rule=escalate_rule, actions=[pagerduty, slack]))
router.add_route(Route(rule=critical_rule, actions=[email]))

# Use with checkpoint
checkpoint = Checkpoint(config, router=router)
```

### Notification Deduplication

Prevent notification spam:

```python
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    InMemoryDeduplicationStore,
    TimeWindow,
    SlidingWindowStrategy,
)

# Create deduplicator with sliding window
deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    default_window=TimeWindow(seconds=3600),  # 1 hour window
)

# Check for duplicates
fingerprint = deduplicator.generate_fingerprint(
    checkpoint_name="data_quality",
    action_type="slack",
    severity="high",
)
if not deduplicator.is_duplicate(fingerprint):
    # Send notification
    deduplicator.mark_sent(fingerprint)
```

### Rate Limiting

```python
from truthound.checkpoint.throttling import (
    ThrottlerBuilder,
    create_throttler,
)

# Build throttler with multiple limits using builder
throttler = (
    ThrottlerBuilder()
    .with_per_minute_limit(10)   # Max 10 per minute
    .with_per_hour_limit(50)     # Max 50 per hour
    .with_per_day_limit(200)     # Max 200 per day
    .build()
)

# Or use create_throttler helper
throttler = create_throttler(
    tokens_per_interval=10,
    interval_seconds=60,
)
```

### Escalation Policies

```python
from truthound.checkpoint.escalation import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicy,
    EscalationPolicyConfig,
    EscalationLevel,
    EscalationTarget,
    TargetType,
    EscalationTrigger,
    create_scheduler,
    create_store,
)

# Define escalation levels
level1 = EscalationLevel(
    level=1,
    delay_seconds=0,
    targets=[
        EscalationTarget(
            type=TargetType.SLACK,
            destination="https://hooks.slack.com/...",
        ),
    ],
)

level2 = EscalationLevel(
    level=2,
    delay_seconds=900,  # 15 minutes
    targets=[
        EscalationTarget(
            type=TargetType.EMAIL,
            destination="oncall@company.com",
        ),
    ],
)

# Create policy
policy = EscalationPolicyConfig(
    name="critical_escalation",
    levels=[level1, level2],
    trigger=EscalationTrigger.ON_FAILURE,
)

# Create escalation engine
engine = EscalationEngine(
    config=EscalationEngineConfig(),
    scheduler=create_scheduler("asyncio"),
    store=create_store("memory"),
)

# Register and trigger
engine.register_policy(policy)
await engine.trigger("critical_escalation", context={"checkpoint_name": "test"})
```

## Storage Backends

### S3 Storage

```python
from truthound.stores.backends.s3 import S3Store

store = S3Store(
    bucket="validation-results",
    prefix="truthound/",
    region="us-west-2",
)

# Save validation result
from truthound.stores import ValidationResult
result = ValidationResult.from_report(report, data_asset="customers.csv")
run_id = store.save(result)

# Retrieve result
stored_result = store.get(run_id)
```

### Database Storage

```python
from truthound.stores.backends.database import DatabaseStore, PoolingConfig

store = DatabaseStore(
    connection_url="postgresql://user:pass@localhost/truthound",
    pooling=PoolingConfig(
        pool_size=10,
        max_overflow=20,
        enable_circuit_breaker=True,
    ),
)

# Save validation result
run_id = store.save(result)
```

### Using Factory Function

```python
from truthound.stores import get_store

# Get filesystem store
fs_store = get_store("filesystem", base_path=".truthound/results")

# Get S3 store
s3_store = get_store("s3", bucket="my-bucket", region="us-west-2")
```

## Monitoring & Metrics

### Prometheus Integration

```python
from truthound.infrastructure.metrics import configure_metrics, get_metrics

# Configure and start HTTP endpoint
configure_metrics(
    enabled=True,
    enable_http=True,
    port=9090,
    service="truthound",
    environment="production",
)

# Get metrics manager
metrics = get_metrics()

# Record validator metrics
with metrics.validator.time("not_null", "users", "email"):
    # Run validation
    pass

# Record checkpoint metrics
metrics.checkpoint.execution_started("daily_check")
metrics.checkpoint.execution_completed("daily_check", success=True, issues=5)

# Metrics available at http://localhost:9090/metrics
# - truthound_validations_total
# - truthound_issues_total
# - truthound_validation_duration_seconds
```

### Dashboard Integration

```python
# Export results for Grafana/DataDog
result = checkpoint.run()

metrics = {
    "validation_passed": result.success,
    "issue_count": len(result.validation_result.issues) if result.validation_result else 0,
    "duration_ms": result.duration_ms,
    "timestamp": result.run_time.isoformat(),
}

# Send to your metrics backend
```

## Environment Configuration

### Using Environment Variables

```python
import os
from truthound.checkpoint import CheckpointConfig

config = CheckpointConfig(
    name="production_validation",
    data_source=os.environ["DATA_SOURCE_PATH"],
)

# Notifications from env
slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
if slack_webhook:
    from truthound.checkpoint.actions import SlackNotification
    checkpoint.add_action(SlackNotification(webhook_url=slack_webhook))
```

### Configuration Profiles

```python
from truthound.infrastructure.config import ConfigManager, ConfigProfile

# Create config manager with environment-specific profiles
manager = ConfigManager()

# Define profiles
dev_profile = ConfigProfile(
    name="development",
    settings={
        "log_level": "DEBUG",
        "storage_backend": "filesystem",
    },
)

prod_profile = ConfigProfile(
    name="production",
    settings={
        "log_level": "INFO",
        "storage_backend": "s3",
    },
)

manager.register_profile(dev_profile)
manager.register_profile(prod_profile)

# Activate profile based on environment
env = os.environ.get("ENVIRONMENT", "development")
manager.activate_profile(env)

# Access settings
print(f"Active profile: {manager.active_profile.name}")
print(f"Settings: {manager.get_all_settings()}")
```

## Pre-commit Hook

Add data validation as a pre-commit hook:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: truthound-check
        name: Truthound Data Validation
        entry: truthound check
        language: system
        files: \.(csv|parquet|json)$
        args: ['--strict', '--min-severity', 'high']
```

## Best Practices

### 1. Start with Baselines

```python
# Create baseline schema before production
schema = th.learn("production_sample.csv", infer_constraints=True)
schema.save("schemas/production_v1.yaml")
```

### 2. Use Severity Filtering

```python
# Development: see all issues
report = th.check(data, min_severity="info")

# Production: only actionable issues
report = th.check(data, min_severity="medium", strict=True)
```

### 3. Version Your Schemas

```yaml
# schemas/sales_v2.yaml
version: "2.0"
created: "2024-01-15"
columns:
  ...
```

### 4. Monitor Trends

```python
# Track metrics over time using AnalyticsService
from truthound.checkpoint.analytics import (
    AnalyticsService,
    InMemoryTimeSeriesStore,
)

# Create analytics service
service = AnalyticsService(store=InMemoryTimeSeriesStore())

# Record checkpoint executions
service.record_execution(
    checkpoint_name="daily_sales",
    success=True,
    duration_ms=1234.5,
    issues=5,
)

# Analyze trends
trend = service.analyze_trend("daily_sales", period_days=30)
print(f"Trend direction: {trend.direction}")
print(f"Trend slope: {trend.slope}")

# Detect anomalies
anomalies = service.detect_anomalies("daily_sales", period_days=7)
for anomaly in anomalies:
    print(f"Anomaly: {anomaly.type} at {anomaly.timestamp}")
```

## Next Steps

- [Data Profiling Tutorial](data-profiling.md) - Learn data characteristics first
- [Custom Validator Tutorial](custom-validator.md) - Build domain-specific validators
- [CI/CD Guide](../guides/ci-cd.md) - Detailed CI/CD configuration
- [Checkpoint Commands](../cli/checkpoint/index.md) - CLI reference
