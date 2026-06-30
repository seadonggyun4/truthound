# truthound 체크포인트 list

List all available 체크포인트 in a 설정 파일.

## Synopsis

```bash
truthound checkpoint list [OPTIONS]
```

## Arguments

CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-c`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 체크포인트 설정 파일 |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

CLI 명령 실행에서 `checkpoint list`을(를) 다루는 항목입니다:

1. **Lists** all 체크포인트 names
2. CLI 명령 실행에서 Shows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. **Displays** 검증기 counts
4. **리포트** notification settings

## 예시

### Basic Usage

```bash
truthound checkpoint list --config truthound.yaml
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Checkpoints (2):
  - daily_data_validation
      Data: data/production.csv
      Actions: 3
      Triggers: 1
  - hourly_metrics_check
      Data: data/metrics.parquet
      Actions: 1
      Triggers: 1
```

### Custom 설정 파일

```bash
truthound checkpoint list --config production.yaml
```

### JSON Output

```bash
truthound checkpoint list --config truthound.yaml --format json
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```json
[
  {
    "name": "daily_data_validation",
    "config": {
      "data_source": "data/production.csv",
      "validators": ["null", "duplicate", "range", "regex"]
    },
    "actions": [...],
    "triggers": [...]
  },
  {
    "name": "hourly_metrics_check",
    "config": {
      "data_source": "data/metrics.parquet",
      "validators": ["null", "range"]
    },
    "actions": [...],
    "triggers": [...]
  }
]
```

CLI 명령 실행에서 JSON, Note을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Detailed View

CLI 명령 실행에서 JSON, `jq`을(를) 다루는 항목입니다:

```bash
truthound checkpoint list --format json | jq '.[] | select(.name == "daily_data_validation")'
```

## Use Cases

### 1. Discovery

Find available 체크포인트 in a project:

```bash
truthound checkpoint list -c truthound.yaml
```

### 2. CI/CD Script

CLI 명령 실행에서 List을(를) 다루는 항목입니다:

```bash
# Run all checkpoints
for checkpoint in $(truthound checkpoint list --format json | jq -r '.[].name'); do
  truthound checkpoint run $checkpoint --strict
done
```

### 3. Documentation

Generate 체크포인트 documentation:

```bash
truthound checkpoint list --format json > docs/checkpoints.json
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- [`checkpoint run`](run.md) - Run a 체크포인트
- [`checkpoint validate`](validate.md) - Validate 설정
- [`checkpoint init`](init.md) - Initialize 설정

## 함께 보기

- [CI/CD 통합 Guide](../../guides/ci-cd.md)
