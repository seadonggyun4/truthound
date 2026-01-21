# Storage Actions

결과 저장 및 문서화를 위한 액션입니다.

## StoreValidationResult

검증 결과를 파일 시스템, S3, GCS 등에 저장합니다.

### 기본 사용법

```python
from truthound.checkpoint.actions import StoreValidationResult

action = StoreValidationResult(
    store_path="./results",
    format="json",
    partition_by="date",
)
```

### 설정 (StoreResultConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `store_path` | `str \| Path` | `"./truthound_results"` | 저장 경로 (로컬, `s3://`, `gs://`) |
| `store_type` | `str` | `"file"` | 저장소 타입: `file`, `s3`, `gcs` |
| `format` | `str` | `"json"` | 포맷: `json`, `yaml` |
| `partition_by` | `str` | `"date"` | 파티션: `date`, `checkpoint`, `status`, `` |
| `retention_days` | `int` | `0` | 보관 기간 (0 = 무제한) |
| `include_validation_details` | `bool` | `True` | 상세 검증 결과 포함 |
| `compress` | `bool` | `False` | gzip 압축 여부 |
| `notify_on` | `str` | `"always"` | 실행 조건 |

### 저장 경로 구조

`partition_by`에 따른 저장 경로:

```
# partition_by="date"
./results/2024/01/15/{run_id}.json

# partition_by="checkpoint"
./results/daily_validation/{run_id}.json

# partition_by="status"
./results/failure/{run_id}.json

# partition_by="" (없음)
./results/{run_id}.json
```

### 로컬 파일 시스템

```python
action = StoreValidationResult(
    store_path="./truthound_results",
    store_type="file",
    format="json",
    partition_by="date",
    compress=True,  # .json.gz로 저장
)
```

### AWS S3

```python
action = StoreValidationResult(
    store_path="s3://my-bucket/dq-results",
    store_type="s3",
    format="json",
    partition_by="date",
)

# AWS 자격 증명은 환경 변수 또는 AWS 설정 파일 사용
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
```

요구 사항: `pip install boto3`

### Google Cloud Storage

```python
action = StoreValidationResult(
    store_path="gs://my-bucket/dq-results",
    store_type="gcs",
    format="json",
    partition_by="checkpoint",
)

# GCP 자격 증명은 GOOGLE_APPLICATION_CREDENTIALS 환경 변수
```

요구 사항: `pip install google-cloud-storage`

### 결과 포맷

저장되는 JSON 구조:

```json
{
  "run_id": "20240115_120000_abc123",
  "checkpoint_name": "daily_validation",
  "run_time": "2024-01-15T12:00:00",
  "status": "failure",
  "data_asset": "users.csv",
  "duration_ms": 1523.5,
  "validation_result": {
    "statistics": {
      "total_issues": 150,
      "critical_issues": 5,
      "high_issues": 25,
      "medium_issues": 70,
      "low_issues": 50,
      "pass_rate": 0.85,
      "total_rows": 100000,
      "total_columns": 15
    },
    "results": [...]  // include_validation_details=True 시
  },
  "action_results": [...],
  "metadata": {...}
}
```

---

## UpdateDataDocs

HTML 형식의 검증 리포트를 생성합니다.

### 기본 사용법

```python
from truthound.checkpoint.actions import UpdateDataDocs

action = UpdateDataDocs(
    site_path="./docs",
    format="html",
    include_history=True,
)
```

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `site_path` | `str \| Path` | `"./truthound_docs"` | 출력 디렉토리 |
| `format` | `str` | `"html"` | 출력 포맷: `html`, `markdown` |
| `include_history` | `bool` | `True` | 히스토리 포함 |
| `max_history_items` | `int` | `100` | 최대 히스토리 개수 |
| `template` | `str` | `"default"` | 템플릿: `default`, `minimal`, `detailed` |
| `notify_on` | `str` | `"always"` | 실행 조건 |

### 생성 파일 구조

```
./docs/
├── index.html                 # 대시보드
├── checkpoints/
│   └── daily_validation/
│       ├── index.html         # 체크포인트 개요
│       └── runs/
│           ├── 20240115_120000.html
│           └── 20240114_120000.html
├── history/
│   └── trend.json            # 트렌드 데이터
└── assets/
    ├── style.css
    └── script.js
```

### 템플릿 옵션

```python
# 기본 템플릿 - 전체 정보 포함
action = UpdateDataDocs(template="default")

# 최소 템플릿 - 핵심 정보만
action = UpdateDataDocs(template="minimal")

# 상세 템플릿 - 모든 이슈 상세 정보
action = UpdateDataDocs(template="detailed")
```

### YAML 설정

```yaml
actions:
  - type: store_result
    store_path: ./truthound_results
    partition_by: date
    format: json
    compress: true

  - type: update_docs
    site_path: ./truthound_docs
    format: html
    include_history: true
    max_history_items: 50
    template: default
```
