# truthound 체크포인트 run

CLI 명령 실행에서 Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound checkpoint run <name> [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Name of the 체크포인트 to run |

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-c`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 설정 파일 (YAML/JSON) |
| CLI 명령 실행에서 `--data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-d`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Override data 소스 path |
| CLI 명령 실행에서 `--validators`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-v`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Override 검증기 (comma-separated) |
| CLI 명령 실행에서 `--output`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 파일 path (JSON) |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--strict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Exit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--store`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 결과 storage directory |
| CLI 명령 실행에서 `--slack`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Slack, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--webhook`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Generic, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--github-summary`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Write, GitHub, Actions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

CLI 명령 실행에서 `checkpoint run`을(를) 다루는 항목입니다:

1. **Loads** the 체크포인트 설정
2. **Validates** the data 자산
3. **Runs** all configured 검증기
4. CLI 명령 실행에서 Sends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. **스토어** 결과 (if configured)

## 예시

### Basic Execution

```bash
truthound checkpoint run daily_data_validation --config truthound.yaml
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Checkpoint: daily_data_validation
=================================
Running validators on data/production.csv...

Validators:
  ✓ null
  ✓ duplicate
  ✗ range (age): 5 values outside [0, 150]
  ✓ regex (email)

Summary:
  Total Validators: 4
  Passed: 3
  Failed: 1
  Status: FAILED
```

### Strict Mode (CI/CD)

CLI 명령 실행에서 Exit을(를) 다루는 항목입니다:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --strict
```

### Override Data 소스

CLI 명령 실행에서 Run을(를) 다루는 항목입니다:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --data /path/to/new/data.csv
```

### Override 검증기

Run only specific 검증기:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --validators null,duplicate,range
```

### JSON Output

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --format json -o results.json
```

Output 파일 (`results.json`):
```json
{
  "checkpoint": "daily_data_validation",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "failed",
  "data_source": "data/production.csv",
  "rows": 10000,
  "results": [
    {
      "validator": "null",
      "passed": true
    },
    {
      "validator": "duplicate",
      "passed": true
    },
    {
      "validator": "range",
      "passed": false,
      "issues": [
        {
          "severity": "high",
          "message": "5 values outside range [0, 150]"
        }
      ]
    },
    {
      "validator": "regex",
      "passed": true
    }
  ],
  "summary": {
    "total_validators": 4,
    "passed": 3,
    "failed": 1
  }
}
```

### Store 결과

Save 결과 to a directory:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --store .truthound/results
```

결과 are stored as:
```
.truthound/results/
└── daily_data_validation/
    └── 2024-01-15T10-30-00/
        ├── report.json
        └── summary.txt
```

### Slack Notification

Send 결과 to Slack:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --slack $SLACK_WEBHOOK_URL
```

CLI 명령 실행에서 YAML을(를) 다루는 항목입니다:
```yaml
checkpoints:
- name: daily_data_validation
  data_source: data/production.csv
  validators:
  - 'null'
  - duplicate
  actions:
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    notify_on: failure
    channel: '#data-quality'
```

### Webhook Notification

Send 결과 to a webhook:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --webhook https://api.example.com/webhook
```

### GitHub Actions Summary

CLI 명령 실행에서 Write, GitHub, Actions을(를) 다루는 항목입니다:

```bash
truthound checkpoint run daily_data_validation -c truthound.yaml --github-summary
```

CLI 명령 실행에서 GitHub, Actions을(를) 다루는 항목입니다:

```markdown
## Data Quality Report: daily_data_validation

| Validator | Status |
|-----------|--------|
| null | Passed |
| duplicate | Passed |
| range | Failed |
| regex | Passed |

### Issues Found

- **range**: 5 values outside range [0, 150]
```

## 설정 파일

### Minimal 설정

```yaml
checkpoints:
- name: my_checkpoint
  data_source: data.csv
  validators:
  - 'null'
```

### Full 설정

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
    run_on_weekdays: [0, 1, 2, 3, 4]

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

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `--strict`, Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `--strict`, Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 환경 변수

| CLI 명령 실행에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| CLI 명령 실행에서 `TRUTHOUND_CONFIG`, TRUTHOUND_CONFIG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Default 설정 파일 path |
| CLI 명령 실행에서 `SLACK_WEBHOOK_URL`, SLACK_WEBHOOK_URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Slack, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `WEBHOOK_URL`, WEBHOOK_URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Generic, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 API, `API_TOKEN`, API_TOKEN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 API을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Use Cases

### 1. CI/CD 파이프라인

```yaml
# GitHub Actions
- name: Run Data Quality Check
  run: |
    truthound checkpoint run daily_data_validation \
      --config truthound.yaml \
      --strict \
      --github-summary
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 2. Scheduled 검증

```bash
# Cron job
0 6 * * * truthound checkpoint run daily_data_validation -c /app/truthound.yaml --store /var/log/truthound
```

### 3. Pre-Deployment Check

```bash
# Before deployment
truthound checkpoint run daily_data_validation -c truthound.yaml --strict || exit 1
```

### 4. Multiple 체크포인트

```bash
# Run multiple checkpoints
for checkpoint in daily_data_validation hourly_metrics_check; do
  truthound checkpoint run $checkpoint -c truthound.yaml --strict
done
```

## Related Commands

- [`checkpoint list`](list.md) - List available 체크포인트
- [`checkpoint validate`](validate.md) - Validate 설정
- [`checkpoint init`](init.md) - Initialize 설정
- [`check`](../core/check.md) - Single 파일 검증

## 함께 보기

- [CI/CD 통합 Guide](../../guides/ci-cd.md)
- [Notification 설정](../../guides/notifications.md)
- CLI 명령 실행에서 Storage, Backends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
