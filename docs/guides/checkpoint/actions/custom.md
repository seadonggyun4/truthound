# Custom Actions

사용자 정의 로직을 실행하는 액션입니다.

## CustomAction

Python 콜백 함수나 셸 명령어를 실행합니다.

### 설정

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `callback` | `Callable \| None` | `None` | Python 콜백 함수 |
| `shell_command` | `str \| None` | `None` | 셸 명령어 |
| `environment` | `dict[str, str]` | `{}` | 환경 변수 |
| `pass_result_as_json` | `bool` | `True` | 결과를 JSON으로 전달 (셸 명령어용) |
| `working_directory` | `str \| None` | `None` | 작업 디렉토리 |
| `notify_on` | `str` | `"always"` | 실행 조건 |

### Python 콜백 사용

```python
from truthound.checkpoint.actions import CustomAction

def my_callback(checkpoint_result):
    """검증 결과를 처리하는 커스텀 로직."""
    status = checkpoint_result.status.value
    stats = checkpoint_result.validation_result.statistics

    print(f"Checkpoint {checkpoint_result.checkpoint_name}: {status}")
    print(f"Total issues: {stats.total_issues}")

    if status == "failure":
        # 커스텀 알림 로직
        send_custom_alert(checkpoint_result)

    # 추가 데이터 저장
    save_to_database(checkpoint_result)

    # 반환값은 ActionResult.details에 포함됨
    return {"processed": True, "custom_metric": 42}


action = CustomAction(
    callback=my_callback,
    notify_on="always",
)
```

### 비동기 콜백

```python
import asyncio

async def async_callback(checkpoint_result):
    """비동기 커스텀 로직."""
    await asyncio.sleep(1)  # 비동기 작업
    await send_notification_async(checkpoint_result)
    return {"async_result": True}


action = CustomAction(callback=async_callback)
```

### 셸 명령어 사용

```python
# 간단한 셸 명령어
action = CustomAction(
    shell_command="./scripts/notify.sh",
    notify_on="failure",
)

# 환경 변수 전달
action = CustomAction(
    shell_command="./scripts/process_result.py",
    environment={
        "API_KEY": "${SECRET_KEY}",
        "ENVIRONMENT": "production",
    },
    pass_result_as_json=True,  # 결과를 stdin으로 전달
    working_directory="./scripts",
)
```

### 셸 스크립트 예시

`pass_result_as_json=True`일 때 결과가 stdin으로 전달됩니다:

```bash
#!/bin/bash
# scripts/process_result.sh

# stdin에서 JSON 읽기
result=$(cat)

# jq로 파싱
status=$(echo $result | jq -r '.status')
issues=$(echo $result | jq -r '.validation_result.statistics.total_issues')
checkpoint=$(echo $result | jq -r '.checkpoint_name')

echo "Checkpoint: $checkpoint"
echo "Status: $status"
echo "Issues: $issues"

# 조건부 처리
if [ "$status" = "failure" ]; then
    curl -X POST "https://api.example.com/alert" \
        -H "Content-Type: application/json" \
        -d "{\"checkpoint\": \"$checkpoint\", \"issues\": $issues}"
fi
```

Python 스크립트 예시:

```python
#!/usr/bin/env python3
# scripts/process_result.py

import json
import sys

# stdin에서 결과 읽기
result = json.load(sys.stdin)

checkpoint = result["checkpoint_name"]
status = result["status"]
stats = result["validation_result"]["statistics"]

print(f"Processing {checkpoint}: {status}")
print(f"Issues: {stats['total_issues']}")

# 커스텀 로직...
```

### 조건부 실행

```python
def conditional_callback(checkpoint_result):
    """특정 조건에서만 실행되는 로직."""
    stats = checkpoint_result.validation_result.statistics

    # Critical 이슈가 10개 이상일 때만 페이지
    if stats.critical_issues >= 10:
        page_on_call_engineer(checkpoint_result)
        return {"paged": True}

    return {"paged": False}


action = CustomAction(
    callback=conditional_callback,
    notify_on="failure",  # 실패 시에만 콜백 호출
)
```

### 에러 처리

콜백에서 예외가 발생하면 ActionResult의 상태가 ERROR가 됩니다:

```python
def risky_callback(checkpoint_result):
    try:
        # 위험한 작업
        result = do_something_risky()
        return {"success": True, "result": result}
    except Exception as e:
        # 예외를 다시 발생시키면 ActionResult.error에 기록됨
        raise RuntimeError(f"Failed to process: {e}")


# 또는 명시적으로 실패 반환
def safe_callback(checkpoint_result):
    try:
        result = do_something_risky()
        return {"success": True}
    except Exception as e:
        # 예외를 잡고 실패 정보 반환
        return {"success": False, "error": str(e)}
```

### 다른 액션과 조합

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.actions import (
    StoreValidationResult,
    SlackNotification,
    CustomAction,
)

def post_process(result):
    """모든 다른 액션이 완료된 후 실행."""
    # 결과 후처리
    aggregate_metrics(result)
    update_dashboard(result)
    return {"post_processed": True}


checkpoint = Checkpoint(
    name="my_check",
    data_source="data.csv",
    validators=["null"],
    actions=[
        # 순서대로 실행됨
        StoreValidationResult(store_path="./results"),      # 1. 저장
        SlackNotification(webhook_url="...", notify_on="failure"),  # 2. 알림
        CustomAction(callback=post_process),                # 3. 후처리
    ],
)
```

---

## YAML 설정 예시

```yaml
actions:
  # 셸 명령어
  - type: custom
    shell_command: ./scripts/notify.sh
    environment:
      API_KEY: ${API_KEY}
    pass_result_as_json: true
    notify_on: failure

  # Python 스크립트
  - type: custom
    shell_command: python ./scripts/process.py
    working_directory: ./scripts
    pass_result_as_json: true
    notify_on: always
```

참고: YAML에서는 Python 콜백을 직접 지정할 수 없습니다. 복잡한 로직이 필요하면 Python API를 사용하세요.
