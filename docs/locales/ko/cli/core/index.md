# Core 명령

CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

| CLI 명령 실행에서 Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Primary, Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|------------------|
| CLI 명령 실행에서 `learn`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Learn 스키마 from data | 스키마 inference |
| CLI 명령 실행에서 `check`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Validate 데이터 품질 | Data 검증 |
| CLI 명령 실행에서 `scan`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Scan, PII을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 개인정보 compliance |
| CLI 명령 실행에서 `mask`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Mask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `profile`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Generate data 프로파일 | CLI 명령 실행에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `read`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Detect data 드리프트 | Model 모니터링 |

## Typical 워크플로우

```mermaid
graph LR
    A[Raw Data] --> B[learn]
    B --> C[schema.yaml]
    A --> D[check]
    C --> D
    D --> E{Issues?}
    E -->|Yes| F[Fix Data]
    E -->|No| G[scan]
    G --> H{PII Found?}
    H -->|Yes| I[mask]
    H -->|No| J[Ready]
    I --> J
```

### 1. 스키마 Learning

CLI 명령 실행에서 First을(를) 다루는 항목입니다:

```bash
truthound learn reference_data.csv -o schema.yaml
```

### 2. Data 검증

CLI 명령 실행에서 Validate을(를) 다루는 항목입니다:

```bash
truthound check new_data.csv --schema schema.yaml --strict
```

### 3. PII Detection

CLI 명령 실행에서 Scan을(를) 다루는 항목입니다:

```bash
truthound scan customer_data.csv
```

### 4. Data Masking

CLI 명령 실행에서 Mask을(를) 다루는 항목입니다:

```bash
truthound mask customer_data.csv -o safe_data.csv --strategy hash
```

### 5. 데이터 프로파일링

CLI 명령 실행에서 Generate을(를) 다루는 항목입니다:

```bash
truthound profile data.csv --format json -o profile.json
```

### 6. 드리프트 Detection

CLI 명령 실행에서 Compare을(를) 다루는 항목입니다:

```bash
truthound compare baseline.csv production.csv --method psi
```

## Common Options

CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

### Output Format (`-f, --format`)

```bash
# Console output (default)
truthound check data.csv

# JSON output
truthound check data.csv --format json

# HTML report
truthound check data.csv --format html -o report.html
```

### Output 파일 (`-o, --output`)

```bash
truthound check data.csv -o results.json --format json
```

### Strict Mode (`--strict`)

CLI 명령 실행에서 Exit, CI/CD을(를) 다루는 항목입니다:

```bash
truthound check data.csv --strict
truthound compare baseline.csv current.csv --strict
```

## Data 소스 Options

CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|-------------|
| CLI 명령 실행에서 `--connection`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `--conn`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 connection string (e.g., `postgresql://user:pass@host/db`) |
| CLI 명령 실행에서 `--table`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | 데이터베이스 테이블 name |
| CLI 명령 실행에서 `--query`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 SQL, `--table`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--source-config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `--sc`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, YAML, Path, JSON/YAML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--source-name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```bash
# Validate a database table directly
truthound check --connection "postgresql://user:pass@host/db" --table users --strict

# Profile from a source config file
truthound profile --source-config prod_db.yaml

# Read and preview database data
truthound read --connection "sqlite:///data.db" --table orders --head 20
```

CLI 명령 실행에서 CLI, Data, Source, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## CI/CD 통합

CLI 명령 실행에서 CI/CD을(를) 다루는 항목입니다:

```yaml
# GitHub Actions example
- name: Validate Data Quality
  run: truthound check data/*.csv --strict

- name: Check for PII
  run: truthound scan data/*.csv --format json -o pii_report.json
```

## 다음 단계

- CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [check](check.md) - Validate 데이터 품질
- CLI 명령 실행에서 Scan, PII을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 Mask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [프로파일](profile.md) - Generate data 프로파일
- CLI 명령 실행에서 Detect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
