# Webhook Actions

HTTP 웹훅을 통해 외부 시스템과 통합하는 액션입니다.

## WebhookAction

모든 HTTP 엔드포인트에 요청을 보낼 수 있는 범용 웹훅 액션입니다.

### 설정 (WebhookConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `url` | `str` | `""` | 웹훅 URL (필수) |
| `method` | `str` | `"POST"` | HTTP 메서드: `GET`, `POST`, `PUT`, `PATCH`, `DELETE` |
| `headers` | `dict[str, str]` | `{}` | 추가 HTTP 헤더 |
| `auth_type` | `str` | `"none"` | 인증 타입: `none`, `basic`, `bearer`, `api_key` |
| `auth_credentials` | `dict[str, str]` | `{}` | 인증 자격 증명 |
| `payload_template` | `dict \| None` | `None` | 커스텀 페이로드 템플릿 |
| `include_full_result` | `bool` | `True` | 전체 결과 포함 여부 |
| `ssl_verify` | `bool` | `True` | SSL 인증서 검증 |
| `success_codes` | `list[int]` | `[200, 201, 202, 204]` | 성공으로 간주할 HTTP 상태 코드 |
| `notify_on` | `str` | `"always"` | 실행 조건 |

### 기본 사용법

```python
from truthound.checkpoint.actions import WebhookAction

# 기본 POST 요청
action = WebhookAction(
    url="https://api.example.com/data-quality/events",
    notify_on="failure",
)

# PUT 요청
action = WebhookAction(
    url="https://api.example.com/status",
    method="PUT",
    notify_on="always",
)
```

### 인증 설정

#### Bearer Token 인증

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="bearer",
    auth_credentials={
        "token": "${API_TOKEN}",  # 환경 변수 참조
    },
)
```

#### Basic 인증

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="basic",
    auth_credentials={
        "username": "user",
        "password": "${API_PASSWORD}",
    },
)
```

#### API Key 인증

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    auth_type="api_key",
    auth_credentials={
        "header": "X-API-Key",  # 헤더 이름 (기본: "X-API-Key")
        "key": "${API_KEY}",
    },
)
```

### 커스텀 헤더

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    headers={
        "X-Custom-Header": "custom-value",
        "X-Request-ID": "${REQUEST_ID}",
        "Accept": "application/json",
    },
)
```

### 커스텀 페이로드

`payload_template`을 사용하여 페이로드를 커스터마이징할 수 있습니다. 플레이스홀더를 지원합니다.

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    payload_template={
        "event_type": "data_quality_check",
        "checkpoint": "${checkpoint}",
        "status": "${status}",
        "run_id": "${run_id}",
        "timestamp": "${run_time}",
        "metrics": {
            "total_issues": "${total_issues}",
            "critical": "${critical_issues}",
            "high": "${high_issues}",
            "pass_rate": "${pass_rate}",
        },
        "custom_field": "custom_value",
    },
)
```

#### 지원 플레이스홀더

| 플레이스홀더 | 설명 |
|--------------|------|
| `${checkpoint}` | 체크포인트 이름 |
| `${run_id}` | 실행 ID |
| `${status}` | 결과 상태 |
| `${run_time}` | 실행 시간 (ISO 8601) |
| `${data_asset}` | 데이터 자산 이름 |
| `${total_issues}` | 총 이슈 수 |
| `${critical_issues}` | Critical 이슈 수 |
| `${high_issues}` | High 이슈 수 |
| `${medium_issues}` | Medium 이슈 수 |
| `${low_issues}` | Low 이슈 수 |
| `${pass_rate}` | 통과율 |

### 기본 페이로드

`payload_template`이 없을 때의 기본 페이로드:

```json
{
  "event": "validation_completed",
  "checkpoint": "daily_validation",
  "run_id": "20240115_120000",
  "status": "failure",
  "run_time": "2024-01-15T12:00:00",
  "data_asset": "users.csv",
  "summary": {
    "total_issues": 150,
    "critical_issues": 5,
    "high_issues": 25,
    "medium_issues": 70,
    "low_issues": 50,
    "pass_rate": 0.85
  },
  "full_result": { ... }  // include_full_result=True 시
}
```

### SSL 검증 비활성화

내부 네트워크의 자체 서명 인증서를 사용하는 경우:

```python
action = WebhookAction(
    url="https://internal.example.com/webhook",
    ssl_verify=False,  # 주의: 보안상 권장하지 않음
)
```

### 성공 코드 커스터마이징

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    success_codes=[200, 201, 202, 204, 302],  # 302 리다이렉트도 성공으로 처리
)
```

### 재시도 설정

```python
action = WebhookAction(
    url="https://api.example.com/webhook",
    timeout_seconds=30,    # 요청 타임아웃
    retry_count=3,         # 실패 시 최대 3회 재시도
    retry_delay_seconds=2, # 재시도 간격 2초
)
```

---

## GitHubAction

GitHub Actions와 통합하는 액션입니다. Job Summary, Annotations, Outputs을 설정합니다.

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `token` | `str` | `""` | GitHub Token (없으면 환경 변수 사용) |
| `repo` | `str` | `""` | Repository (owner/repo) |
| `check_name` | `str` | `"Truthound"` | Check Run 이름 |
| `step_summary` | `bool` | `True` | Job Summary 작성 |
| `set_output` | `bool` | `True` | Workflow 출력 설정 |
| `annotations` | `bool` | `True` | 오류/경고 Annotation 출력 |
| `notify_on` | `str` | `"always"` | 실행 조건 |

### 사용 예시

```python
from truthound.checkpoint.actions import GitHubAction

action = GitHubAction(
    token="${GITHUB_TOKEN}",
    repo="owner/repo",
    step_summary=True,
    set_output=True,
    annotations=True,
)
```

### GitHub Actions Workflow에서 사용

```yaml
- name: Run Data Quality Check
  run: truthound checkpoint run my_check --config config.yaml
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

- name: Use Outputs
  run: |
    echo "Status: ${{ steps.dq-check.outputs.status }}"
    echo "Issues: ${{ steps.dq-check.outputs.total_issues }}"
```

---

## YAML 설정 예시

```yaml
actions:
  # 기본 웹훅
  - type: webhook
    url: https://api.example.com/data-quality/events
    method: POST
    notify_on: failure

  # 인증 설정
  - type: webhook
    url: https://api.example.com/webhook
    method: POST
    auth_type: bearer
    auth_credentials:
      token: ${API_TOKEN}
    headers:
      X-Custom-Header: custom-value
    notify_on: always

  # 커스텀 페이로드
  - type: webhook
    url: https://api.example.com/webhook
    payload_template:
      event: data_quality
      checkpoint: "${checkpoint}"
      status: "${status}"
      issues: "${total_issues}"
    include_full_result: false
    notify_on: failure_or_error

  # GitHub Actions 통합
  - type: github
    step_summary: true
    set_output: true
    annotations: true
    notify_on: always
```
