# truthound 체크포인트 init

CLI 명령 실행에서 Initialize을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound checkpoint init [OPTIONS]
```

## Arguments

CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--output`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `truthound.yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 파일 path |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설정 format (yaml, json) |

## Description

CLI 명령 실행에서 `checkpoint init`을(를) 다루는 항목입니다:

1. **Creates** a well-documented 설정 template
2. CLI 명령 실행에서 Includes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. CLI 명령 실행에서 Shows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. CLI 명령 실행에서 Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시

### Basic Initialization

```bash
truthound checkpoint init
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Sample checkpoint config created: truthound.yaml

Edit the file to configure your checkpoints, then run:
  truthound checkpoint run <checkpoint_name> --config truthound.yaml
```

### Custom Output Path

```bash
truthound checkpoint init -o config/data-quality.yaml
```

### JSON Format

```bash
truthound checkpoint init --format json -o truthound.json
```

## Generated 설정

### YAML Output (default)

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
      # Column-specific range constraints
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
    run_on_weekdays:
    - 0
    - 1
    - 2
    - 3
    - 4

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

### JSON Output

```json
{
  "checkpoints": [
    {
      "name": "daily_data_validation",
      "data_source": "data/production.csv",
      "validators": ["null", "duplicate", "range"],
      "validator_config": {
        "range": {
          "columns": {
            "age": {"min_value": 0, "max_value": 150},
            "price": {"min_value": 0}
          }
        }
      },
      "min_severity": "medium",
      "auto_schema": true,
      "tags": {
        "environment": "production",
        "team": "data-platform"
      },
      "actions": [
        {
          "type": "store_result",
          "store_path": "./truthound_results",
          "partition_by": "date"
        },
        {
          "type": "update_docs",
          "site_path": "./truthound_docs",
          "include_history": true
        },
        {
          "type": "slack",
          "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
          "notify_on": "failure",
          "channel": "#data-quality"
        }
      ],
      "triggers": [
        {
          "type": "schedule",
          "interval_hours": 24,
          "run_on_weekdays": [0, 1, 2, 3, 4]
        }
      ]
    },
    {
      "name": "hourly_metrics_check",
      "data_source": "data/metrics.parquet",
      "validators": ["null", "range"],
      "validator_config": {
        "range": {
          "columns": {
            "value": {"min_value": 0, "max_value": 100},
            "count": {"min_value": 0}
          }
        }
      },
      "actions": [
        {
          "type": "webhook",
          "url": "https://api.example.com/data-quality/events",
          "auth_type": "bearer",
          "auth_credentials": {"token": "${API_TOKEN}"}
        }
      ],
      "triggers": [
        {
          "type": "cron",
          "expression": "0 * * * *"
        }
      ]
    }
  ]
}
```

## 설정 Sections

### 체크포인트 Definition

```yaml
checkpoints:
- name: my_checkpoint           # Required: unique identifier
  data_source: path/to/file.csv # Required: data file path
  validators:                   # Required: list of validators
  - 'null'
  - duplicate
```

### 검증기

```yaml
validators:
- 'null'        # Check for null values
- duplicate     # Check for duplicates
- range         # Check numeric ranges
```

!!! note "참고"
CLI 명령 실행에서 YAML, `pattern`, RegexValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
CLI 명령 실행에서 API, Python을(를) 다루는 항목입니다:
    ```python
    from truthound.validators import RegexValidator
    th.check(data, validators=[RegexValidator(pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$", columns=["email"])])
    ```

### 검증기 설정

```yaml
validator_config:
  range:
    columns:
      age:
        min_value: 0
        max_value: 150
```

### Actions

```yaml
actions:
- type: store_result          # Store validation results
  store_path: ./results
  partition_by: date

- type: update_docs           # Generate HTML documentation
  site_path: ./docs
  include_history: true

- type: slack                 # Slack notification
  webhook_url: ${SLACK_WEBHOOK_URL}
  notify_on: failure
  channel: '#data-quality'

- type: webhook               # Generic webhook
  url: https://api.example.com/webhook
  auth_type: bearer
  auth_credentials:
    token: ${API_TOKEN}
```

### Triggers

```yaml
triggers:
- type: schedule              # Time-interval based
  interval_hours: 24
  run_on_weekdays: [0, 1, 2, 3, 4]

- type: cron                  # Cron expression
  expression: "0 * * * *"     # Every hour
```

## Use Cases

### 1. New Project Setup

```bash
# Initialize new project
mkdir my-data-project && cd my-data-project
truthound checkpoint init
```

### 2. 빠른 시작

```bash
# Initialize, validate, and run
truthound checkpoint init

# Edit truthound.yaml to configure your data source...

# Run a checkpoint
truthound checkpoint run daily_data_validation --config truthound.yaml
```

### 3. CI/CD Template

```bash
# Generate CI/CD-ready configuration
truthound checkpoint init -o .github/truthound.yaml
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- [`checkpoint validate`](validate.md) - Validate 설정
- [`checkpoint run`](run.md) - Run a 체크포인트
- [`checkpoint list`](list.md) - List 체크포인트

## 함께 보기

- [시작하기 Guide](../../getting-started/quickstart.md)
- [CI/CD 통합](../../guides/ci-cd.md)
- [설정 레퍼런스](../../guides/configuration.md)
