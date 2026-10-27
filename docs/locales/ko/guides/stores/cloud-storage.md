# Cloud Storage Backends

실무 운영 가이드에서 SQL, Enterprise, AWS, Google, Cloud, Storage, Azure, Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설치

```bash
# AWS S3
pip install truthound[s3]

# Google Cloud Storage
pip install truthound[gcs]

# Azure Blob Storage
pip install truthound[azure]

# SQL Database (SQLAlchemy)
pip install truthound[db]

# All cloud backends
pip install truthound[all-stores]
```

## S3 Store

실무 운영 가이드에서 `boto3`, AWS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.stores.backends.s3 import S3Store

store = S3Store(
    bucket="my-validation-bucket",
    prefix="validations/",
    region="us-east-1",
)

# Save and retrieve
run_id = store.save(result)
result = store.get(run_id)
```

### Using the Factory

```python
from truthound.stores import get_store

store = get_store(
    "s3",
    bucket="my-bucket",
    prefix="validations/",
    region="us-east-1",
)
```

### 설정

```python
from truthound.stores.backends.s3 import S3Config

config = S3Config(
    bucket="my-bucket",                    # S3 bucket name
    prefix="truthound/",                   # Key prefix
    namespace="default",                   # Namespace for organization
    region="us-east-1",                    # AWS region
    endpoint_url=None,                     # Custom endpoint (MinIO, LocalStack)
    use_compression=True,                  # Gzip compression
    storage_class="STANDARD",              # S3 storage class
    server_side_encryption="AES256",       # SSE setting
    kms_key_id=None,                       # KMS key for aws:kms
    tags={"env": "prod"},                  # Object tags
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `bucket`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `prefix`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Key을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `region`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 AWS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `endpoint_url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Custom, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `use_compression`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Compress을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `storage_class`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `STANDARD`, STANDARD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `server_side_encryption`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SSE, AES256을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `kms_key_id`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 KMS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `tags`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict[str, str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Object을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 조건부 게시와 복구

`S3Store(config=S3Config(...))`는 중첩 metadata·object tag까지 입력 설정을
복사합니다. 개별 생성자 설정도 JSON·safe YAML(`serialization_format`)과
`compression="zstd"` 같은 네이티브 압축 이름을 지원합니다. `True`는 gzip,
`False`는 무압축이며 pickle·알 수 없는 설정은 provider 작업 전에 거부합니다.
기본 metadata보다 각 결과의 metadata가 우선하고 object tag의 이름·값은 모두
URL 인코딩합니다.

초기화는 index와 실제 결과 목록·ID·metadata를 대조합니다. 새 index에는
SHA-256과 불투명 ETag receipt도 저장합니다. hash가 없는 구형 index는 결과와
일치할 때 읽을 수 있습니다. 읽기는 네이티브 response body를 닫습니다.
`max_object_bytes`(기본 128 MiB)는 인코딩된 object·index를 제한하고 복원 결과
크기도 검사합니다. 복원 크기 검사 전에 codec이 메모리를 할당할 수 있습니다.
query 오류는 숨기지 않고 `limit=0`은 빈 결과입니다. `run_time`·`data_asset`·
`status` 정렬을 적용합니다. provider의 not-found만 부재로 처리하며 접근 거부·
연결 실패는 오류입니다.

결과·index는 확인된 ETag로 조건부 게시합니다. 새 key는 `IfNoneMatch="*"`,
교체는 이전 `IfMatch` receipt를 사용합니다. 삭제는 결과 ETag와 HTTP 204를
확인합니다. 이미 오래된 소유자는 바뀐 index를 확인하면 다음 결과 쓰기 전에
차단됩니다. 실패·불명확한 쓰기는 후속 변경을 막습니다. 결과·index는 별도 object
작업이므로 동시 경쟁은 불명확한 object를 남길 수 있으며, 여러 object의 원자적
transaction이나 독점 원격 lease를 보장하지 않습니다. namespace당 활성 writer를
하나로 유지하고 불명확한 상태는 재사용 전에 조정하십시오.

`rebuild_index()`는 명시적인 복구이며 손상·불일치 index로 초기화가 실패한
경우에도 실행합니다. 현재 형식의 모든 결과를 검증한 뒤 index를 조건부 교체합니다.
잘못된 결과를 건너뛰지 않습니다. 재구축은 index에 없던 유효 결과를 포함한 현재
목록을 채택하고 없어진 결과의 항목을 제거하므로 namespace를 확인한 뒤
선택하십시오. close는 소유 네이티브 client를 닫으며 실패한 handle은 명시적
재시도를 위해 보존합니다. 승인된 MCP S3 팩토리도 같은 네이티브 경로와
`rebuild_index`를 사용하고 provider key·tag·비공개 metadata는 클라이언트
결과에서 숨깁니다. S3 호환 서비스는 [조건부 쓰기](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html)·
[삭제](https://docs.aws.amazon.com/en_en/AmazonS3/latest/API/API_DeleteObject.html)
계약을 지원해야 합니다. 현재 개발 검증은 실제 botocore 요청 모델과 재시작 후
복원되는 합성 object를 사용했으며 실서비스는 별도 미검증입니다.

### S3-Compatible Services

```python
# MinIO
store = S3Store(
    bucket="my-bucket",
    endpoint_url="http://localhost:9000",
    region="us-east-1",
)

# LocalStack
store = S3Store(
    bucket="my-bucket",
    endpoint_url="http://localhost:4566",
    region="us-east-1",
)
```

### Server-Side Encryption

```python
# SSE-S3
store = S3Store(
    bucket="my-bucket",
    server_side_encryption="AES256",
)

# SSE-KMS
store = S3Store(
    bucket="my-bucket",
    server_side_encryption="aws:kms",
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/abc123",
)
```

## GCS Store

실무 운영 가이드에서 `google-cloud-storage`, Google, Cloud, Storage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.stores.backends.gcs import GCSStore

store = GCSStore(
    bucket="my-validation-bucket",
    prefix="validations/",
    project="my-gcp-project",
)

run_id = store.save(result)
result = store.get(run_id)
```

### 설정

```python
from truthound.stores.backends.gcs import GCSConfig

config = GCSConfig(
    bucket="my-bucket",                    # GCS bucket name
    prefix="truthound/",                   # Object prefix
    namespace="default",                   # Namespace
    project="my-project",                  # GCP project ID
    credentials_path=None,                 # Service account JSON path
    use_compression=True,                  # Gzip compression
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `bucket`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 GCS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `prefix`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Object을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `project`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 GCP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `credentials_path`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, Service을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `use_compression`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Compress을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Generation receipt와 복구

`GCSStore(config=GCSConfig(...))`는 중첩 결과 metadata까지 입력 설정을
복사합니다. 개별 설정은 `serialization_format="json"`·`"yaml"`과
`compression="zstd"` 같은 네이티브 codec 이름을 받습니다. `True`는 gzip,
`False`는 무압축이며 알 수 없는 설정·pickle은 거부합니다. 각 결과의 metadata가
설정 기본값보다 우선하고 SDK 연결은 작업 시점까지 지연합니다.

GCS는 S3와 같은 Core 결과/index·query·체크섬·명시적 재구축 로직을 사용합니다.
새 게시에는 `if_generation_match=0`, 교체·index 갱신·삭제에는 이전의 정확한
generation을 사용합니다. media 읽기 전에 metadata를 새로 읽고
`raw_download=True`와 고정 generation으로 원본 bytes를 유지하여 Google HTTP
decoder와 Core가 gzip을 중복 복원하지 않게 합니다. generation receipt가 없으면
index에 확정하지 않고 후속 변경을 차단합니다. 읽기 실패·손상 index는 오류이며
접근 거부를 부재로 처리하지 않습니다. `rebuild_index()`는 index에 없던 결과를
포함한 현재 유효한 목록을 명시적으로 검증·채택한 뒤 index를 조건부 교체합니다.
`blob_name`을 쓰던 구형 index는 hash·generation이 없어도 ID·metadata가 일치하면
읽을 수 있으며 손상 결과를 건너뛰지 않습니다.

`max_object_bytes` 기본값은 128 MiB이며 인코딩된 object·index와 복원 결과를
검사하지만 codec heap을 격리하지는 않습니다. query 정렬·`limit=0`은 공통
object 저장소 계약을 따릅니다. close는 실제 네이티브 client를 종료하며 실패한
handle은 명시적 재시도를 위해 보존합니다. namespace당 writer를 하나로 유지해야
합니다. 결과·index는 별도 작업으로 불명확한 상태를 남길 수 있으며 다중 object
transaction·독점 원격 lease가 아닙니다. 재사용 전에 조정하십시오. 승인된 MCP
팩토리도 같은 네이티브 수명주기·재구축을 제공하고 provider key·project/자격증명
정보·비공개 metadata는 클라이언트 결과에서 숨깁니다. 개발 검증은 소유한 영속
HTTP transport에 실제 Google SDK의 JSON/media 직렬화·gzip decoder를 연결하고
설치형 worker 재시작까지 확인했으며 실서비스 검증과는 별개입니다.
[Generation 전제조건](https://docs.cloud.google.com/python/docs/reference/storage/latest/generation_metageneration)·
[Blob API](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob)를 참고하십시오.

### Authentication

```python
# Default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
store = GCSStore(bucket="my-bucket")

# Explicit service account
store = GCSStore(
    bucket="my-bucket",
    credentials_path="/path/to/service-account.json",
)

# With project ID
store = GCSStore(
    bucket="my-bucket",
    project="my-gcp-project",
)
```

## Azure Blob Store

실무 운영 가이드에서 `azure-storage-blob`, Azure, Blob, Storage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.stores.backends.azure_blob import AzureBlobStore

store = AzureBlobStore(
    container="my-container",
    connection_string="DefaultEndpointsProtocol=https;...",
)

run_id = store.save(result)
result = store.get(run_id)
```

### 설정

```python
from truthound.stores.backends.azure_blob import AzureBlobConfig

config = AzureBlobConfig(
    container="my-container",              # Container name
    prefix="truthound/",                   # Blob name prefix
    namespace="default",                   # Namespace
    connection_string=None,                # Connection string
    account_url=None,                      # Account URL
    account_name=None,                     # Account name
    account_key=None,                      # Account key
    sas_token=None,                        # SAS token
    use_compression=True,                  # Gzip compression
    content_type="application/json",       # Content type
    access_tier="Hot",                     # Access tier
    metadata={"env": "prod"},              # Blob metadata
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `container`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Container을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `prefix`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `connection_string`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `account_url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Account, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `account_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Account을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `account_key`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Account을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sas_token`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SAS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `use_compression`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Compress을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `access_tier`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Hot, Cool, 아카이브 |
| 실무 운영 가이드에서 `metadata`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `dict[str, str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `{}`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Authentication Methods

```python
# Connection string (recommended for development)
store = AzureBlobStore(
    container="my-container",
    connection_string="DefaultEndpointsProtocol=https;...",
)

# Account URL with SAS token
store = AzureBlobStore(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
    sas_token="sv=2021-06-08&ss=b&srt=sco&sp=rwdlacx&...",
)

# Account name and key
store = AzureBlobStore(
    container="my-container",
    account_name="myaccount",
    account_key="...",
)

# Managed identity (DefaultAzureCredential)
store = AzureBlobStore(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
)
```

### Access Tiers

```python
store = AzureBlobStore(
    container="my-container",
    connection_string="...",
    access_tier="Cool",  # Hot, Cool, Cold, Archive
)

# Change tier for existing result
store.set_access_tier("run-123", "Archive")
```

## 데이터베이스 Store

실무 운영 가이드에서 PostgreSQL, SQLite, MySQL, SQL, SQLAlchemy, Supports, SQLAlchemy-compatible을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.stores.backends.database import DatabaseStore

# SQLite
store = DatabaseStore(
    connection_url="sqlite:///validations.db",
)

# PostgreSQL
store = DatabaseStore(
    connection_url="postgresql://user:pass@localhost/validations",
)

run_id = store.save(result)
result = store.get(run_id)
```

### 설정

```python
from truthound.stores.backends.database import DatabaseConfig, PoolingConfig

config = DatabaseConfig(
    connection_url="postgresql://user:pass@localhost/db",
    namespace="default",                   # Namespace
    table_prefix="",                       # Table name prefix
    pool_size=5,                           # Connection pool size
    max_overflow=10,                       # Max overflow connections
    echo=False,                            # Echo SQL statements
    create_tables=True,                    # Auto-create tables
    use_pool_manager=True,                 # Use enterprise pool manager
    pooling=PoolingConfig(                 # Advanced pooling
        pool_size=5,
        max_overflow=10,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
        enable_circuit_breaker=True,
        enable_health_checks=True,
        enable_retry=True,
        max_retries=3,
        retry_base_delay=0.1,
    ),
)
```

#### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `connection_url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `sqlite:///.truthound/store.db`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SQL, SQLAlchemy, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `table_prefix`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 테이블 name prefix |
| 실무 운영 가이드에서 `pool_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `5`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_overflow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `10`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `echo`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SQL, Echo을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `create_tables`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Auto-create 테이블 |
| 실무 운영 가이드에서 `use_pool_manager`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Connection URLs

```python
# SQLite
DatabaseStore(connection_url="sqlite:///data.db")
DatabaseStore(connection_url="sqlite:///:memory:")

# PostgreSQL
DatabaseStore(connection_url="postgresql://user:pass@host:5432/db")
DatabaseStore(connection_url="postgresql+psycopg2://user:pass@host/db")

# MySQL
DatabaseStore(connection_url="mysql+pymysql://user:pass@host:3306/db")

# SQL Server
DatabaseStore(connection_url="mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17")
```

### Enterprise Pooling

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound.stores.backends.database import DatabaseStore, PoolingConfig

store = DatabaseStore(
    connection_url="postgresql://user:pass@localhost/db",
    pooling=PoolingConfig(
        pool_size=10,
        max_overflow=20,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
        enable_circuit_breaker=True,
        enable_health_checks=True,
        enable_retry=True,
        max_retries=3,
    ),
)

# Check health
if store.is_healthy:
    print("Database connection is healthy")

# Get pool metrics
metrics = store.pool_metrics
if metrics:
    print(f"Active connections: {metrics.active_connections}")

# Get comprehensive status
status = store.get_pool_status()

# Manual connection recycling
recycled = store.recycle_connections()
```

### 재시도 with Exponential Backoff

```python
from sqlalchemy import text

def insert_and_query(session):
    session.execute(text("INSERT INTO my_table ..."))
    return session.execute(text("SELECT * FROM my_table")).fetchall()

# Automatic retry on transient errors
result = store.execute_with_retry(insert_and_query)
```

## Common Features

### Compression

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
# Enabled (default for cloud stores)
store = S3Store(bucket="my-bucket", compression=True)

# Disabled
store = S3Store(bucket="my-bucket", compression=False)
```

### Namespace Organization

```python
# Development namespace
dev_store = S3Store(bucket="my-bucket", namespace="development")

# Production namespace
prod_store = S3Store(bucket="my-bucket", namespace="production")
```

### Error Handling

```python
from truthound.stores.base import (
    StoreConnectionError,
    StoreNotFoundError,
    StoreReadError,
    StoreWriteError,
)

try:
    store.initialize()
except StoreConnectionError as e:
    print(f"Connection failed: {e}")

try:
    result = store.get("nonexistent")
except StoreNotFoundError as e:
    print(f"Not found: {e.item_id}")
```

## Backend Comparison

| 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 GCS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Azure, Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 |
|---------|-----|-----|------------|----------|
| 실무 운영 가이드에서 Compression을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Server-side을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 N/A을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Access을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 N/A을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tags을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 N/A을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 N/A을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 N/A을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Circuit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 재시도 with backoff | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 다음 단계

- 실무 운영 가이드에서 FileSystem, Store, Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Versioning, Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Tiering, Hot/Warm/Cold/Archive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Caching, In-memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

S3와 GCS도 종료에 실패한 handle을 유지하고 명시적인 close 재시도가 성공할
때까지 새 IO를 거부합니다. 종료가 확인되지 않은 client를 정상 상태로 재사용하지 않습니다.
