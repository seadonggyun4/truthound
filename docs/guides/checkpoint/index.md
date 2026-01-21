# Checkpoint & CI/CD Integration (Phase 6)

Truthound's Checkpoint system provides a comprehensive framework for orchestrating automated data quality validation pipelines with enterprise-grade CI/CD integration.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Actions](#actions)
5. [Triggers](#triggers)
6. [Async Execution](#async-execution)
7. [Transaction Management](#transaction-management)
8. [CI/CD Integration](#cicd-integration)
9. [CI Reporters](#ci-reporters)
10. [CheckpointRunner](#checkpointrunner)
11. [Registry](#registry)
12. [Advanced Notifications](#advanced-notifications)
13. [Best Practices](#best-practices)
14. [API Reference](#api-reference)
15. [Enterprise Assessment](#enterprise-assessment)

---

## Overview

Checkpoints combine data sources, validators, actions, and triggers into reusable validation pipelines that can be run manually or automatically. The system provides:

- **Automated Validation Pipelines**: Define once, run anywhere
- **12 CI/CD Platform Support**: Native integration with major CI systems
- **Async Execution**: Non-blocking, high-throughput validation
- **Transaction Management**: Saga pattern with compensation and idempotency
- **Flexible Triggers**: Schedule, Cron, Event, and File-based triggers

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Checkpoint                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ DataSource  │  │ Validators  │  │   Actions   │  │  Triggers  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┼────────────────┼────────────────┘        │
│                          ▼                ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    CheckpointRunner                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ ││
│  │  │   Sync      │  │   Async     │  │  Transaction Coordinator │ ││
│  │  │  Execution  │  │  Execution  │  │  (Saga + Idempotency)    │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     CI/CD Reporters                              ││
│  │  GitHub │ GitLab │ Jenkins │ CircleCI │ Azure │ Bitbucket │ ... ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.actions import (
    StoreValidationResult,
    SlackNotification,
)

# Create a checkpoint
checkpoint = Checkpoint(
    name="daily_user_validation",
    data_source="users.csv",
    validators=["null", "duplicate", "range"],
    actions=[
        StoreValidationResult(store_path="./results"),
        SlackNotification(
            webhook_url="https://hooks.slack.com/...",
            notify_on="failure",
            channel="#data-quality"
        ),
    ],
)

# Run the checkpoint
result = checkpoint.run()
print(result.summary())
```

### CLI Usage

```bash
# Initialize a sample configuration
truthound checkpoint init -o truthound.yaml

# Run a checkpoint from config
truthound checkpoint run daily_data_validation --config truthound.yaml

# Run ad-hoc checkpoint
truthound checkpoint run quick_check --data data.csv --validators null,duplicate

# List checkpoints
truthound checkpoint list --config truthound.yaml

# Validate configuration
truthound checkpoint validate truthound.yaml
```

---

## Core Components

### CheckpointConfig

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig

config = CheckpointConfig(
    name="production_validation",
    data_source="s3://bucket/data.parquet",
    validators=["null", "duplicate", "range"],
    min_severity="medium",
    schema="schema.yaml",
    auto_schema=False,
    run_name_template="%Y%m%d_%H%M%S",
    tags={"env": "production", "team": "data-platform"},
    metadata={"owner": "data-team@company.com"},
    fail_on_critical=True,
    fail_on_high=False,
    timeout_seconds=3600,
    sample_size=100000,
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
  - regex
  validator_config:
    regex:
      patterns:
        email: ^[\w.+-]+@[\w-]+\.[\w.-]+$
        product_code: ^[A-Z]{2,4}[-_][0-9]{3,6}$
        phone: ^(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$
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
    run_on_weekdays: [0, 1, 2, 3, 4]  # Mon-Fri
```

---

## Actions

Actions are executed after validation completes. They can store results, send notifications, or integrate with external systems.

### Available Actions

| Action | Description | Key Features |
|--------|-------------|--------------|
| `StoreValidationResult` | Save results to filesystem, S3, or GCS | Partitioning, compression, retention |
| `UpdateDataDocs` | Generate HTML/Markdown documentation | History tracking, templates |
| `SlackNotification` | Send Slack messages via webhook | Mentions, custom formatting |
| `EmailNotification` | Send email notifications | SMTP, SendGrid, SES support |
| `WebhookAction` | Call any HTTP endpoint | Auth types, custom headers |
| `PagerDutyAction` | Create/resolve PagerDuty incidents | Auto-resolve on success |
| `GitHubAction` | GitHub Actions integration | Summaries, outputs, annotations |
| `TeamsNotification` | Microsoft Teams notifications | Adaptive Cards, themes, templates |
| `OpsGenieAction` | OpsGenie alert management | Priorities, responders, escalation |
| `DiscordNotification` | Discord webhook notifications | Embeds, mentions |
| `TelegramNotification` | Telegram bot notifications | Markdown, photos |
| `CustomAction` | Execute Python callbacks or shell commands | Full flexibility |

### StoreValidationResult

```python
from truthound.checkpoint.actions import StoreValidationResult

action = StoreValidationResult(
    store_path="./results",      # Local path, s3://, or gs://
    store_type="file",           # file, s3, gcs
    format="json",               # json, yaml
    partition_by="date",         # date, checkpoint, status
    retention_days=30,
    compress=True,
)
```

### UpdateDataDocs

```python
from truthound.checkpoint.actions import UpdateDataDocs

action = UpdateDataDocs(
    site_path="./docs",
    format="html",               # html, markdown
    include_history=True,
    max_history_items=100,
    template="default",          # default, minimal, detailed
)
```

### SlackNotification

```python
from truthound.checkpoint.actions import SlackNotification

action = SlackNotification(
    webhook_url="https://hooks.slack.com/...",
    channel="#data-quality",
    notify_on="failure",         # always, success, failure, error
    mention_on_failure=["U12345", "@here"],
    include_details=True,
    custom_message="Data quality check completed",
)
```

### WebhookAction

```python
from truthound.checkpoint.actions import WebhookAction

action = WebhookAction(
    url="https://api.example.com/webhook",
    method="POST",
    auth_type="bearer",          # none, basic, bearer, api_key
    auth_credentials={"token": "${API_TOKEN}"},
    headers={"X-Custom-Header": "value"},
    include_full_result=True,
    timeout_seconds=30,
    retry_count=3,
)
```

### TeamsNotification

Microsoft Teams notifications with Adaptive Cards:

```python
from truthound.checkpoint.actions import (
    TeamsNotification,
    TeamsConfig,
    AdaptiveCardBuilder,
    MessageTheme,
    create_teams_notification,
    create_failure_alert,
)

# Basic usage
action = TeamsNotification(
    webhook_url="https://outlook.office.com/webhook/...",
    notify_on="failure",
    channel="Data Quality",
    include_details=True,
)

# With custom Adaptive Card
builder = AdaptiveCardBuilder()
builder.add_header("Data Quality Alert")
builder.add_fact("Dataset", "users.csv")
builder.add_fact("Issues", "150")
builder.add_action_button("View Report", "https://...")

action = TeamsNotification(
    webhook_url="...",
    card_builder=builder,
    theme=MessageTheme.CRITICAL,
)

# Factory functions
action = create_failure_alert(
    webhook_url="...",
    mention_users=["user@company.com"],
)
```

### OpsGenieAction

OpsGenie alert management with priorities and responders:

```python
from truthound.checkpoint.actions import (
    OpsGenieAction,
    OpsGenieConfig,
    AlertPriority,
    ResponderType,
    Responder,
    create_opsgenie_action,
    create_critical_alert,
    create_team_alert,
)

# Basic usage
action = OpsGenieAction(
    api_key="${OPSGENIE_API_KEY}",
    notify_on="failure",
    priority=AlertPriority.P1,
    tags=["data-quality", "production"],
)

# With responders
action = OpsGenieAction(
    api_key="...",
    responders=[
        Responder(type=ResponderType.TEAM, name="data-platform"),
        Responder(type=ResponderType.USER, username="oncall@company.com"),
    ],
    visible_to=[
        Responder(type=ResponderType.TEAM, name="engineering"),
    ],
    auto_resolve_on_success=True,
)

# Factory functions
action = create_critical_alert(
    api_key="...",
    team="data-platform",
    escalation_policy="data-quality-escalation",
)
```

### DiscordNotification

Discord webhook notifications:

```python
from truthound.checkpoint.actions import DiscordNotification, DiscordConfig

action = DiscordNotification(
    webhook_url="https://discord.com/api/webhooks/...",
    notify_on="failure",
    username="Truthound Bot",
    avatar_url="https://example.com/logo.png",
    embed_color=0xFF0000,  # Red for errors
    include_mentions=["@here"],
)

# With custom embed
action = DiscordNotification(
    webhook_url="...",
    embed_title="Data Quality Alert",
    embed_description="Validation failed for users.csv",
    embed_fields=[
        {"name": "Issues", "value": "150", "inline": True},
        {"name": "Severity", "value": "High", "inline": True},
    ],
)
```

### TelegramNotification

Telegram bot notifications:

```python
from truthound.checkpoint.actions import (
    TelegramNotification,
    TelegramConfig,
    TelegramNotificationWithPhoto,
)

# Basic text notification
action = TelegramNotification(
    bot_token="${TELEGRAM_BOT_TOKEN}",
    chat_id="-1001234567890",  # Channel/group ID
    notify_on="failure",
    parse_mode="Markdown",  # or "HTML"
)

# With photo (e.g., chart screenshot)
action = TelegramNotificationWithPhoto(
    bot_token="...",
    chat_id="...",
    photo_url="https://example.com/chart.png",
    caption="Data quality trend chart",
)

# Custom message template
action = TelegramNotification(
    bot_token="...",
    chat_id="...",
    message_template="""
🚨 *Data Quality Alert*

Dataset: `{checkpoint_name}`
Status: {status}
Issues: {issue_count}

[View Report]({report_url})
""",
)
```

### CustomAction

```python
from truthound.checkpoint.actions import CustomAction

# Python callback
def my_callback(result):
    print(f"Checkpoint completed: {result.status}")
    if result.status == "failure":
        # Custom alerting logic
        send_custom_alert(result)
    return {"processed": True}

action = CustomAction(callback=my_callback)

# Shell command
action = CustomAction(
    shell_command="./scripts/notify.sh",
    environment={"API_KEY": "${SECRET_KEY}"},
    pass_result_as_json=True,
)
```

### Notify Conditions

All actions support the `notify_on` parameter:

| Condition | Triggers On |
|-----------|-------------|
| `always` | Every run |
| `success` | Validation passed |
| `failure` | Validation failed |
| `error` | System error occurred |
| `failure_or_error` | Failure or error |
| `not_success` | Any non-success status |

---

## Triggers

Triggers determine when checkpoints should run automatically.

### ScheduleTrigger

Time-interval based execution:

```python
from truthound.checkpoint.triggers import ScheduleTrigger

# Run every hour
trigger = ScheduleTrigger(interval_hours=1)

# Run every 30 minutes on weekdays
trigger = ScheduleTrigger(
    interval_minutes=30,
    run_on_weekdays=[0, 1, 2, 3, 4],  # Mon=0, Sun=6
    start_time=datetime(2024, 1, 1, 9, 0),  # Start at 9 AM
    end_time=datetime(2024, 12, 31, 18, 0),  # End at 6 PM
    timezone="America/New_York",
)
```

### CronTrigger

Standard cron expression support (5 or 6 fields):

```python
from truthound.checkpoint.triggers import CronTrigger

# Daily at midnight
trigger = CronTrigger(expression="0 0 * * *")

# Every 15 minutes
trigger = CronTrigger(expression="*/15 * * * *")

# Monday at 9am
trigger = CronTrigger(expression="0 9 * * 1")

# With seconds (6 fields)
trigger = CronTrigger(expression="30 0 9 * * 1")  # Monday 9:00:30
```

### EventTrigger

Event-driven execution with filtering and debouncing:

```python
from truthound.checkpoint.triggers import EventTrigger

trigger = EventTrigger(
    event_type="data_updated",
    event_filter={"source": "production", "priority": "high"},
    debounce_seconds=60,       # Minimum time between triggers
    batch_events=True,         # Batch multiple events
    batch_window_seconds=30,   # Batch window
)

# Fire event programmatically
trigger.fire_event({
    "source": "production",
    "priority": "high",
    "table": "users",
    "rows_affected": 1500,
})
```

### FileWatchTrigger

File system change detection with hash-based verification:

```python
from truthound.checkpoint.triggers import FileWatchTrigger

trigger = FileWatchTrigger(
    paths=["./data", "/shared/datasets"],
    patterns=["*.csv", "*.parquet"],
    recursive=True,
    events=["modified", "created"],  # modified, created, deleted
    ignore_patterns=[".*", "__pycache__", "*.tmp"],
    hash_check=True,           # Only trigger on content change
    poll_interval_seconds=5,
)
```

---

## Async Execution

For high-throughput scenarios, use `AsyncCheckpoint` for non-blocking execution:

```python
import asyncio
from truthound.checkpoint import AsyncCheckpoint
from truthound.checkpoint.async_actions import AsyncSlackNotification

# Create async checkpoint
checkpoint = AsyncCheckpoint(
    name="async_validation",
    data_source="large_dataset.parquet",
    validators=["null", "duplicate"],
    actions=[
        AsyncSlackNotification(webhook_url="..."),
    ],
    max_concurrent_actions=5,
    execution_strategy="concurrent",  # sequential, concurrent, pipeline
)

# Run asynchronously
async def main():
    result = await checkpoint.run_async()
    print(result.summary())

asyncio.run(main())
```

### Execution Strategies

```python
from truthound.checkpoint.async_base import (
    SequentialStrategy,
    ConcurrentStrategy,
    PipelineStrategy,
)

# Sequential: One action at a time
checkpoint = AsyncCheckpoint(
    execution_strategy=SequentialStrategy()
)

# Concurrent: All actions in parallel with limit
checkpoint = AsyncCheckpoint(
    execution_strategy=ConcurrentStrategy(max_concurrency=10)
)

# Pipeline: Staged execution
# Stage 1: Store result and update docs (parallel)
# Stage 2: Notify (after stage 1)
checkpoint = AsyncCheckpoint(
    execution_strategy=PipelineStrategy(
        stages=[[0, 1], [2]]  # Action indices
    )
)
```

### Running Multiple Checkpoints Concurrently

```python
from truthound.checkpoint import run_checkpoints_async

checkpoints = [checkpoint1, checkpoint2, checkpoint3]

results = await run_checkpoints_async(
    checkpoints,
    max_concurrent=5,
    context={"batch_id": "2024-01-15"},
)

for result in results:
    print(f"{result.checkpoint_name}: {result.status}")
```

### Async Callbacks

```python
async def on_complete(result):
    await send_metrics_async(result.to_dict())

async def on_error(result):
    await alert_team_async(result.error)

checkpoint = AsyncCheckpoint(
    name="monitored_check",
    on_complete=on_complete,
    on_error=on_error,
)
```

---

## Transaction Management

Truthound provides enterprise-grade transaction support with the Saga pattern:

### Compensatable Actions

Actions that can be rolled back on failure:

```python
from truthound.checkpoint.transaction import CompensatableAction

class DatabaseUpdateAction(CompensatableAction):
    def execute(self, result):
        # Forward action
        self.backup_id = create_backup()
        update_database(result)
        return ActionResult(status="success")

    def compensate(self, result, execute_result):
        # Rollback action
        restore_from_backup(self.backup_id)
        return ActionResult(status="compensated")
```

### Transaction Coordinator

```python
from truthound.checkpoint.transaction import TransactionCoordinator

coordinator = TransactionCoordinator(
    actions=[action1, action2, action3],
    compensation_strategy="reverse",  # reverse, parallel
    max_compensation_retries=3,
)

result = coordinator.execute(checkpoint_result)

if result.needs_rollback:
    coordinator.compensate(result)
```

### Idempotency

Prevent duplicate executions:

```python
from truthound.checkpoint.idempotency import IdempotencyService

service = IdempotencyService(
    store="redis://localhost:6379",  # Or filesystem, memory
    ttl_seconds=3600,
)

# Check before execution
idempotency_key = f"checkpoint:{name}:{run_id}"

if service.is_duplicate(idempotency_key):
    return cached_result

result = checkpoint.run()
service.mark_completed(idempotency_key, result)
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on:
  schedule:
    - cron: '0 0 * * *'
  push:
    paths:
      - 'data/**'
  pull_request:
    paths:
      - 'data/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Truthound
        run: pip install truthound[all]

      - name: Run Validation
        run: |
          truthound checkpoint run daily_data_validation \
            --config truthound.yaml \
            --github-summary \
            --strict
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: data-quality-report
          path: truthound_results/
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - validate

data-quality:
  stage: validate
  image: python:3.11-slim
  variables:
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"
  cache:
    paths:
      - .pip-cache/
  script:
    - pip install truthound[all]
    - truthound checkpoint run $CHECKPOINT_NAME --config truthound.yaml
  artifacts:
    when: always
    paths:
      - truthound_results/
      - truthound_docs/
    reports:
      dotenv: truthound.env
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - data/**/*
```

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        SLACK_WEBHOOK = credentials('slack-webhook')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install truthound[all]'
            }
        }

        stage('Data Quality') {
            steps {
                sh '''
                    truthound checkpoint run daily_data_validation \
                        --config truthound.yaml \
                        --format json \
                        --output truthound-result.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'truthound-result.json'
                    junit 'truthound-junit.xml'
                }
                failure {
                    slackSend channel: '#data-quality',
                              message: "Data Quality Check Failed: ${env.BUILD_URL}"
                }
            }
        }
    }
}
```

### CircleCI

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  data-quality:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - run:
          name: Install Dependencies
          command: pip install truthound[all]
      - run:
          name: Run Validation
          command: |
            truthound checkpoint run daily_data_validation \
              --config truthound.yaml \
              --format json
      - store_test_results:
          path: test-results/truthound
      - store_artifacts:
          path: artifacts

workflows:
  nightly:
    triggers:
      - schedule:
          cron: "0 0 * * *"
          filters:
            branches:
              only: main
    jobs:
      - data-quality
```

### Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  paths:
    include:
      - data/*

schedules:
  - cron: "0 0 * * *"
    displayName: Daily midnight run
    branches:
      include:
        - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: pip install truthound[all]
    displayName: 'Install Truthound'

  - script: |
      truthound checkpoint run $(CHECKPOINT_NAME) \
        --config truthound.yaml \
        --format json
    displayName: 'Run Data Quality Check'
    env:
      SLACK_WEBHOOK: $(SLACK_WEBHOOK)

  - publish: truthound_results
    artifact: DataQualityReport
    condition: always()
```

### Generate CI Configs

```python
from truthound.checkpoint.ci import (
    generate_github_workflow,
    generate_gitlab_ci,
    generate_jenkinsfile,
    generate_circleci_config,
    write_ci_config,
)

# Generate GitHub Actions workflow
workflow = generate_github_workflow(
    checkpoint_name="daily_data_validation",
    schedule="0 0 * * *",
    notify_slack=True,
    python_version="3.11",
)

# Generate all configs
write_ci_config("github", checkpoint_name="daily_data_validation")
write_ci_config("gitlab", checkpoint_name="daily_data_validation")
write_ci_config("jenkins", checkpoint_name="daily_data_validation")
```

---

## CI Reporters

Truthound automatically detects CI environments and provides platform-specific reporters:

### Supported Platforms

| Platform | Detection | Features |
|----------|-----------|----------|
| **GitHub Actions** | `GITHUB_ACTIONS=true` | Step Summary, Annotations, Outputs |
| **GitLab CI** | `GITLAB_CI=true` | dotenv artifacts, ANSI colors |
| **Jenkins** | `JENKINS_URL` | JUnit XML, Properties file |
| **CircleCI** | `CIRCLECI=true` | test-results, Artifacts |
| **Azure DevOps** | `TF_BUILD=True` | Build Variables |
| **Bitbucket Pipelines** | `BITBUCKET_BUILD_NUMBER` | Pipes compatible |
| **Travis CI** | `TRAVIS=true` | Environment mapping |
| **TeamCity** | `TEAMCITY_VERSION` | Service messages |
| **Buildkite** | `BUILDKITE=true` | Annotations API |
| **Drone** | `DRONE=true` | Environment variables |
| **AWS CodeBuild** | `CODEBUILD_BUILD_ID` | BuildSpec compatible |
| **GCP Cloud Build** | `BUILDER_OUTPUT` | Environment detection |

### Using CI Reporters

```python
from truthound.checkpoint.ci import (
    detect_ci_platform,
    get_ci_environment,
    get_ci_reporter,
    is_ci_environment,
)

# Check if in CI
if is_ci_environment():
    env = get_ci_environment()
    print(f"Platform: {env.platform}")
    print(f"Repository: {env.repository}")
    print(f"Branch: {env.branch}")
    print(f"Commit: {env.commit_sha}")
    print(f"PR Number: {env.pr_number}")
    print(f"Run URL: {env.run_url}")

# Get platform-specific reporter
reporter = get_ci_reporter()
reporter.report_status(result)
reporter.set_output("total_issues", stats.total_issues)
reporter.set_output("status", result.status.value)
```

### GitHub Actions Reporter

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter(
    step_summary=True,      # Write to GITHUB_STEP_SUMMARY
    use_groups=True,        # Use ::group:: for collapsible sections
    emoji_enabled=True,     # Include emojis in output
    set_output=True,        # Set workflow outputs
)

# Report to GitHub
exit_code = reporter.report_to_ci(result)

# The reporter automatically:
# - Writes job summary in Markdown
# - Emits annotations (::error::, ::warning::, ::notice::)
# - Sets output variables via GITHUB_OUTPUT
```

### Custom Annotations

```python
from truthound.reporters.ci.base import CIAnnotation, AnnotationLevel

annotation = CIAnnotation(
    message="Null values exceed threshold (15% > 5%)",
    level=AnnotationLevel.ERROR,
    file="data/users.csv",
    line=42,
    title="Data Quality Issue",
    validator_name="NullValidator",
)

reporter.format_annotation(annotation)
# Output: ::error file=data/users.csv,line=42,title=Data Quality Issue::Null values exceed threshold (15% > 5%)
```

---

## CheckpointRunner

The runner manages automated execution of checkpoints with triggers:

```python
from truthound.checkpoint import Checkpoint, CheckpointRunner
from truthound.checkpoint.triggers import ScheduleTrigger, CronTrigger

# Create checkpoints with triggers
hourly_metrics_check = Checkpoint(
    name="hourly_metrics_check",
    data_source="data.csv",
    validators=["null", "duplicate"],
).add_trigger(ScheduleTrigger(interval_hours=1))

daily_data_validation = Checkpoint(
    name="daily_data_validation",
    data_source="data.parquet",
    validators=["range", "distribution"],
).add_trigger(CronTrigger(expression="0 0 * * *"))

# Create runner
runner = CheckpointRunner(
    max_workers=4,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
    error_callback=lambda e: print(f"Error: {e}"),
)

# Add checkpoints
runner.add_checkpoint(hourly_metrics_check)
runner.add_checkpoint(daily_data_validation)

# Start background execution
runner.start()

# Run specific checkpoint once
result = runner.run_once("hourly_metrics_check")

# Run all checkpoints
results = runner.run_all()

# Iterate over results
for result in runner.iter_results(timeout=60):
    print(result.summary())

# Stop runner
runner.stop()
```

---

## Registry

Register checkpoints for global access:

```python
from truthound.checkpoint import (
    Checkpoint,
    CheckpointRegistry,
    register_checkpoint,
    get_checkpoint,
    list_checkpoints,
    load_checkpoints,
)

# Create registry
registry = CheckpointRegistry()

# Register checkpoints
checkpoint = Checkpoint(name="my_check", data_source="data.csv")
registry.register(checkpoint)

# Or use global registry
register_checkpoint(checkpoint)

# Retrieve
cp = get_checkpoint("my_check")
result = cp.run()

# List all
names = list_checkpoints()
print(names)  # ['my_check', ...]

# Load from file
checkpoints = load_checkpoints("truthound.yaml")
for cp in checkpoints:
    registry.register(cp)

# Check existence
if "my_check" in registry:
    print("Checkpoint exists")
```

---

## Advanced Notifications

Truthound provides enterprise-grade notification management including routing, deduplication, throttling, and escalation.

### Rule-based Routing

Route notifications to different actions based on conditions:

```python
from truthound.checkpoint.routing import ActionRouter, SeverityRule, Route
from truthound.checkpoint.actions import SlackNotification, PagerDutyAction

router = ActionRouter()

# Critical alerts go to PagerDuty
router.add_route(Route(
    name="critical",
    rule=SeverityRule(min_severity="critical"),
    actions=[PagerDutyAction(service_key="...")],
    priority=1,
))

# High severity goes to Slack
router.add_route(Route(
    name="high",
    rule=SeverityRule(min_severity="high"),
    actions=[SlackNotification(webhook_url="...")],
    priority=2,
))

# Use with checkpoint
checkpoint = Checkpoint(
    name="daily_data_validation",
    data_source="data.csv",
    router=router,
)
```

**Available Rules**: SeverityRule, IssueCountRule, StatusRule, TagRule, PassRateRule, TimeWindowRule, DataAssetRule, MetadataRule, ErrorRule, AlwaysRule, NeverRule

**Combinators**: `AllOf`, `AnyOf`, `NotRule` for complex conditions.

### Notification Deduplication

Prevent duplicate notifications within a time window:

```python
from truthound.checkpoint.deduplication import (
    NotificationDeduplicator,
    InMemoryDeduplicationStore,
    TimeWindow,
)

deduplicator = NotificationDeduplicator(
    store=InMemoryDeduplicationStore(),
    default_window=TimeWindow(seconds=300),  # 5 minutes
)

fingerprint = deduplicator.generate_fingerprint(
    checkpoint_name="daily_data_validation",
    action_type="slack",
    severity="high",
)

if not deduplicator.is_duplicate(fingerprint):
    await action.execute(result)
    deduplicator.mark_sent(fingerprint)
```

**Window Strategies**: Sliding, Tumbling, Session, Adaptive

**Storage Backends**: InMemory, Redis Streams

### Rate Limiting / Throttling

Control notification frequency:

```python
from truthound.checkpoint.throttling import ThrottlerBuilder, ThrottlingMiddleware

throttler = (
    ThrottlerBuilder()
    .with_per_minute_limit(10)
    .with_per_hour_limit(100)
    .with_per_day_limit(500)
    .build()
)

middleware = ThrottlingMiddleware(throttler=throttler)
throttled_action = middleware.wrap(slack_action)
```

**Algorithms**: Token Bucket, Fixed Window, Sliding Window, Composite

### Escalation Policies

Multi-level alert escalation:

```python
from truthound.checkpoint.escalation import (
    EscalationPolicy,
    EscalationLevel,
    EscalationEngine,
)

policy = EscalationPolicy(
    name="critical_alerts",
    levels=[
        EscalationLevel(level=1, delay_minutes=0, targets=["team-lead"]),
        EscalationLevel(level=2, delay_minutes=15, targets=["manager"]),
        EscalationLevel(level=3, delay_minutes=30, targets=["director"]),
    ],
)

engine = EscalationEngine(policy=policy)
await engine.trigger("incident-123", context={"severity": "critical"})

# Later: acknowledge or resolve
await engine.acknowledge("incident-123", acknowledged_by="john@company.com")
await engine.resolve("incident-123", resolved_by="jane@company.com")
```

**States**: PENDING → TRIGGERED → ACKNOWLEDGED → ESCALATED → RESOLVED

**Storage Backends**: InMemory, Redis, SQLite

---

## Best Practices

### 1. Use Configuration Files

Store checkpoint definitions in version-controlled YAML files:

```yaml
# truthound.yaml
checkpoints:
  - name: production_daily
    data_source: ${DATA_PATH}  # Use environment variables
    validators:
      - "null"
      - duplicate
    actions:
      - type: store_result
        store_path: ${RESULTS_PATH}
```

### 2. Set Up Appropriate Notifications

```python
actions = [
    # Always store results for audit
    StoreValidationResult(notify_on="always"),

    # Update docs on success
    UpdateDataDocs(notify_on="success"),

    # Alert only on failures
    SlackNotification(notify_on="failure"),
    PagerDutyAction(notify_on="failure_or_error"),
]
```

### 3. Use Strict Mode in CI

```bash
truthound checkpoint run my_check --strict
```

In strict mode (`--strict`), exit code 1 is returned if:
- Any validation issues are found (regardless of severity)
- The checkpoint status is "failure" or "error"

This ensures CI pipeline fails appropriately on any data quality issues.

### 4. Leverage Async for Large Datasets

```python
# For large datasets, use async execution
checkpoint = AsyncCheckpoint(
    name="large_data_check",
    data_source="large_dataset.parquet",
    sample_size=100000,  # Sample for faster validation
    max_concurrent_actions=10,
)

result = await checkpoint.run_async()
```

### 5. Implement Idempotency for Production

```python
from truthound.checkpoint.idempotency import IdempotencyService

service = IdempotencyService(store="redis://localhost:6379")

# Prevent duplicate runs
if not service.is_duplicate(run_key):
    result = checkpoint.run()
    service.mark_completed(run_key, result)
```

### 6. Monitor Trends Over Time

```python
StoreValidationResult(
    store_path="s3://bucket/dq-results",
    partition_by="date",
    retention_days=90,
)
```

---

## API Reference

### CheckpointResult

```python
result = checkpoint.run()

result.run_id              # Unique run identifier
result.checkpoint_name     # Checkpoint name
result.run_time           # When the checkpoint ran
result.status             # CheckpointStatus (success/failure/error/warning)
result.validation_result  # ValidationResult from check()
result.action_results     # List of ActionResult
result.duration_ms        # Execution duration in milliseconds
result.error              # Error message if failed
result.metadata           # Custom metadata dict

# Methods
result.to_dict()          # Serialize to dictionary
result.from_dict(d)       # Deserialize from dictionary
result.summary()          # Human-readable summary string
```

### CheckpointStatus

```python
from truthound.checkpoint.checkpoint import CheckpointStatus

CheckpointStatus.SUCCESS    # All validations passed
CheckpointStatus.FAILURE    # Validation failures detected
CheckpointStatus.WARNING    # Non-critical issues found
CheckpointStatus.ERROR      # System error occurred
```

### ActionResult

```python
from truthound.checkpoint.actions.base import ActionResult, ActionStatus

result = ActionResult(
    action_name="slack_notification",
    action_type="notification",
    status=ActionStatus.SUCCESS,
    message="Notification sent successfully",
    started_at=datetime.now(),
    completed_at=datetime.now(),
    duration_ms=150.5,
    details={"message_id": "abc123"},
)
```

### CIEnvironment

```python
from truthound.checkpoint.ci import get_ci_environment

env = get_ci_environment()

env.platform        # CIPlatform enum
env.is_ci           # bool
env.is_pr           # bool (is pull request)
env.branch          # str
env.commit_sha      # str
env.commit_message  # str
env.pr_number       # int | None
env.pr_target_branch # str
env.repository      # str (owner/repo)
env.run_id          # str
env.run_url         # str
env.actor           # str (user who triggered)
env.job_name        # str
env.workflow_name   # str
```

---

## Enterprise Assessment

### Feature Completeness

| Feature | Status | Notes |
|---------|--------|-------|
| Core Checkpoint | ✅ | Full implementation |
| Multiple Actions | ✅ | 12 built-in actions |
| 4 Trigger Types | ✅ | Schedule, Cron, Event, FileWatch |
| Async Execution | ✅ | Native async/await |
| Transaction Management | ✅ | Saga pattern |
| Idempotency | ✅ | Duplicate prevention |
| 12 CI Platforms | ✅ | Industry-leading coverage |
| JUnit XML Output | ✅ | For Jenkins/CI |
| Rule-based Routing | ✅ | 11 rules, combinators, Python/Jinja2 engine |
| Notification Deduplication | ✅ | InMemory/Redis, 4 window strategies |
| Rate Limiting | ✅ | Token Bucket, 5 throttler types |
| Escalation Policies | ✅ | State machine, 3 storage backends |

### Code Metrics

| Metric | Value |
|--------|-------|
| **Total LOC** | ~25,000 |
| **Test LOC** | ~4,600 |
| **Test Files** | 5 |
| **CI Platforms** | 12 |
| **Action Types** | 12 |
| **Trigger Types** | 4 |

### Comparison with Great Expectations

| Feature | Great Expectations | Truthound |
|---------|-------------------|-----------|
| Checkpoint Definition | ✅ | ✅ |
| Multiple Actions | ✅ | ✅ |
| Schedule Triggers | ✅ | ✅ (+ Cron) |
| Event Triggers | ⚠️ Limited | ✅ Full |
| File Watch Triggers | ❌ | ✅ |
| Async Execution | ❌ | ✅ Native |
| Transaction/Saga | ❌ | ✅ |
| Idempotency | ❌ | ✅ |
| CI Platforms | 3-4 | 12 |
| JUnit Output | Plugin | ✅ Built-in |
| Rule-based Routing | ❌ | ✅ 11 rules |
| Deduplication | ❌ | ✅ InMemory/Redis |
| Rate Limiting | ❌ | ✅ Token Bucket |
| Escalation Policies | ❌ | ✅ APScheduler |

---

## See Also

- [Data Sources](DATASOURCES.md) - Connecting to various data backends
- [Validators Reference](VALIDATORS.md) - 289 validators reference
- [Storage Backends](STORES.md) - Storing validation results
- [Reporters](REPORTERS.md) - Output formats and customization
- [Examples](EXAMPLES.md) - Complete usage examples
