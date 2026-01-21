# Checkpoint 기본 사용법

Checkpoint는 데이터 소스, Validators, Actions를 하나의 실행 단위로 묶어 자동화된 데이터 품질 검증 파이프라인을 구성합니다.

## Checkpoint 생성

### Python API

```python
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.checkpoint.actions import StoreValidationResult, SlackNotification

# 기본 생성
checkpoint = Checkpoint(
    name="daily_user_validation",
    data_source="users.csv",
    validators=["null", "duplicate", "range"],
)

# Config 객체 사용
config = CheckpointConfig(
    name="production_validation",
    data_source="s3://bucket/data.parquet",
    validators=["null", "duplicate", "range"],
    validator_config={
        "range": {"column": "age", "min": 0, "max": 120}
    },
    min_severity="medium",
    fail_on_critical=True,
    fail_on_high=False,
    timeout_seconds=3600,
    sample_size=100000,
    tags={"env": "production", "team": "data-platform"},
    metadata={"owner": "data-team@company.com"},
)

checkpoint = Checkpoint(config=config)
```

### YAML 설정

```yaml
# truthound.yaml
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
    min_severity: medium
    auto_schema: true
    tags:
      environment: production
      team: data-platform
    actions:
      - type: store_result
        store_path: ./truthound_results
        partition_by: date
      - type: slack
        webhook_url: https://hooks.slack.com/services/...
        notify_on: failure
```

## CheckpointConfig 속성

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | `"default_checkpoint"` | 체크포인트 고유 이름 |
| `data_source` | `str \| Any` | `""` | 데이터 소스 경로 또는 객체 |
| `validators` | `list[str \| Validator]` | `None` | 실행할 validator 목록 |
| `validator_config` | `dict` | `{}` | validator별 설정 |
| `min_severity` | `str` | `None` | 최소 severity 필터 (`critical`, `high`, `medium`, `low`) |
| `schema` | `str` | `None` | 스키마 파일 경로 |
| `auto_schema` | `bool` | `False` | 스키마 자동 추론 여부 |
| `run_name_template` | `str` | `"%Y%m%d_%H%M%S"` | run_id 생성 템플릿 |
| `fail_on_critical` | `bool` | `True` | critical 이슈 시 실패 처리 |
| `fail_on_high` | `bool` | `False` | high 이슈 시 실패 처리 |
| `timeout_seconds` | `int` | `3600` | 실행 타임아웃 (초) |
| `sample_size` | `int` | `None` | 샘플링 크기 (None = 전체) |
| `tags` | `dict[str, str]` | `{}` | 태그 (라우팅, 필터링용) |
| `metadata` | `dict[str, Any]` | `{}` | 메타데이터 |

## Checkpoint 실행

### 동기 실행

```python
# 단일 실행
result = checkpoint.run()

# 결과 확인
print(result.status)           # CheckpointStatus.SUCCESS/FAILURE/ERROR/WARNING
print(result.run_id)           # 고유 실행 ID
print(result.duration_ms)      # 실행 시간 (ms)
print(result.summary())        # 요약 문자열

# 검증 결과 접근
validation = result.validation_result
print(validation.statistics.total_issues)
print(validation.statistics.pass_rate)

# 액션 결과 확인
for action_result in result.action_results:
    print(f"{action_result.action_name}: {action_result.status}")
```

### CLI 실행

```bash
# YAML 설정 파일에서 체크포인트 실행
truthound checkpoint run daily_data_validation --config truthound.yaml

# Ad-hoc 실행
truthound checkpoint run quick_check \
    --data data.csv \
    --validators null,duplicate

# 엄격 모드 (이슈 발견 시 exit code 1)
truthound checkpoint run my_check --config truthound.yaml --strict

# JSON 출력
truthound checkpoint run my_check --format json --output result.json

# GitHub Actions 요약 포함
truthound checkpoint run my_check --github-summary
```

## CheckpointStatus

실행 결과의 상태입니다.

```python
from truthound.checkpoint import CheckpointStatus

class CheckpointStatus(str, Enum):
    SUCCESS = "success"    # 모든 검증 통과
    FAILURE = "failure"    # Critical/High 이슈 발견 (fail_on_* 설정에 따름)
    ERROR = "error"        # 실행 중 오류 발생
    WARNING = "warning"    # 이슈 발견되었으나 허용 범위
    RUNNING = "running"    # 실행 중
    PENDING = "pending"    # 대기 중
```

### 상태 결정 로직

```python
# CheckpointResult 상태는 다음 로직으로 결정됨
def determine_status(validation_result, config):
    stats = validation_result.statistics

    # 실행 오류가 있으면 ERROR
    if validation_result.error:
        return CheckpointStatus.ERROR

    # Critical 이슈 + fail_on_critical=True면 FAILURE
    if config.fail_on_critical and stats.critical_issues > 0:
        return CheckpointStatus.FAILURE

    # High 이슈 + fail_on_high=True면 FAILURE
    if config.fail_on_high and stats.high_issues > 0:
        return CheckpointStatus.FAILURE

    # 이슈가 있으면 WARNING
    if stats.total_issues > 0:
        return CheckpointStatus.WARNING

    return CheckpointStatus.SUCCESS
```

## CheckpointResult

실행 결과를 담는 데이터클래스입니다.

```python
@dataclass
class CheckpointResult:
    run_id: str                              # 고유 실행 ID
    checkpoint_name: str                     # 체크포인트 이름
    run_time: datetime                       # 실행 시작 시간
    status: CheckpointStatus                 # 결과 상태
    validation_result: ValidationResult      # 검증 결과 객체
    action_results: list[ActionResult]       # 액션 실행 결과 리스트
    data_asset: str                          # 검증된 데이터 자산 이름
    duration_ms: float                       # 총 소요 시간 (밀리초)
    error: str | None                        # 에러 메시지 (에러 시)
    metadata: dict[str, Any]                 # 사용자 메타데이터
```

### 결과 직렬화

```python
# Dictionary로 변환
data = result.to_dict()

# JSON 저장
import json
with open("result.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

# Dictionary에서 복원
restored = CheckpointResult.from_dict(data)
```

## Actions 추가

```python
from truthound.checkpoint.actions import (
    StoreValidationResult,
    SlackNotification,
    WebhookAction,
)

checkpoint = Checkpoint(
    name="with_actions",
    data_source="data.csv",
    validators=["null"],
    actions=[
        # 항상 결과 저장
        StoreValidationResult(
            store_path="./results",
            partition_by="date",
            notify_on="always",
        ),
        # 실패 시 Slack 알림
        SlackNotification(
            webhook_url="https://hooks.slack.com/services/...",
            channel="#data-quality",
            notify_on="failure",
        ),
        # 웹훅 호출
        WebhookAction(
            url="https://api.example.com/webhook",
            method="POST",
            notify_on="failure_or_error",
        ),
    ],
)
```

## Triggers 추가

자동 실행을 위한 트리거를 설정합니다.

```python
from truthound.checkpoint.triggers import ScheduleTrigger, CronTrigger

# 1시간마다 실행
checkpoint = Checkpoint(
    name="hourly_check",
    data_source="data.csv",
    validators=["null"],
)
checkpoint.add_trigger(ScheduleTrigger(interval_hours=1))

# Cron 표현식 사용
checkpoint.add_trigger(CronTrigger(expression="0 9 * * 1"))  # 매주 월요일 9시
```

## CheckpointRunner

여러 체크포인트를 자동 실행합니다.

```python
from truthound.checkpoint import CheckpointRunner

runner = CheckpointRunner(
    max_workers=4,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
    error_callback=lambda e: print(f"Error: {e}"),
)

# 체크포인트 추가
runner.add_checkpoint(checkpoint1)
runner.add_checkpoint(checkpoint2)

# 백그라운드 실행 시작 (트리거 기반)
runner.start()

# 특정 체크포인트 1회 실행
result = runner.run_once("checkpoint1")

# 모든 체크포인트 실행
results = runner.run_all()

# 결과 이터레이션
for result in runner.iter_results(timeout=1.0):
    print(result.summary())

# 종료
runner.stop()
```

## Registry

체크포인트를 전역 레지스트리에 등록하여 이름으로 접근합니다.

```python
from truthound.checkpoint import (
    CheckpointRegistry,
    register_checkpoint,
    get_checkpoint,
    list_checkpoints,
    load_checkpoints,
)

# 전역 레지스트리 사용
register_checkpoint(checkpoint)

# 이름으로 조회
cp = get_checkpoint("my_check")
result = cp.run()

# 목록 조회
names = list_checkpoints()  # ['my_check', ...]

# YAML에서 로드
checkpoints = load_checkpoints("truthound.yaml")
for cp in checkpoints:
    register_checkpoint(cp)

# 커스텀 레지스트리
registry = CheckpointRegistry()
registry.register(checkpoint)

if "my_check" in registry:
    cp = registry.get("my_check")
```

## 다음 단계

- [Actions 상세](./actions/index.md) - 14개 액션 타입 설명
- [Triggers 상세](./triggers.md) - 4개 트리거 타입
- [Routing](./routing.md) - Rule-based 액션 라우팅
- [Async Execution](./async.md) - 비동기 실행
