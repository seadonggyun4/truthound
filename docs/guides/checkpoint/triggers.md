# Triggers

Triggers는 Checkpoint의 자동 실행 시점을 결정합니다. 시간 기반, Cron 표현식, 이벤트 기반, 파일 감시 방식을 지원합니다.

## 트리거 기본 클래스

모든 트리거는 `BaseTrigger[ConfigT]`를 상속합니다.

```python
# src/truthound/checkpoint/triggers/base.py

class TriggerStatus(str, Enum):
    """트리거 상태."""
    ACTIVE = "active"      # 활성화
    PAUSED = "paused"      # 일시 중지
    STOPPED = "stopped"    # 중지됨
    ERROR = "error"        # 오류


@dataclass
class TriggerConfig:
    """트리거 기본 설정."""
    name: str | None = None           # 트리거 이름
    enabled: bool = True              # 활성화 여부
    max_runs: int = 0                 # 최대 실행 횟수 (0 = 무제한)
    run_immediately: bool = False     # 시작 시 즉시 실행
    catch_up: bool = False            # 누락된 실행 보충
    max_concurrent: int = 1           # 최대 동시 실행 수
    metadata: dict = field(default_factory=dict)


@dataclass
class TriggerResult:
    """트리거 평가 결과."""
    should_run: bool          # 실행 여부
    reason: str = ""          # 이유
    next_run: datetime | None = None  # 다음 실행 시간
    context: dict = field(default_factory=dict)  # 추가 컨텍스트
```

---

## ScheduleTrigger

시간 간격 기반 트리거입니다.

### 설정 (ScheduleConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `interval_seconds` | `int` | `0` | 간격 (초) |
| `interval_minutes` | `int` | `0` | 간격 (분) |
| `interval_hours` | `int` | `0` | 간격 (시간) |
| `start_time` | `datetime \| None` | `None` | 시작 시간 (None = 즉시) |
| `end_time` | `datetime \| None` | `None` | 종료 시간 (None = 무제한) |
| `run_on_weekdays` | `list[int] \| None` | `None` | 실행 요일 (0=월, 6=일) |
| `timezone` | `str \| None` | `None` | 타임존 |

### 사용 예시

```python
from truthound.checkpoint.triggers import ScheduleTrigger
from datetime import datetime

# 1시간마다 실행
trigger = ScheduleTrigger(interval_hours=1)

# 30분마다 실행
trigger = ScheduleTrigger(interval_minutes=30)

# 평일(월-금) 업무 시간에만 실행
trigger = ScheduleTrigger(
    interval_minutes=30,
    run_on_weekdays=[0, 1, 2, 3, 4],  # 월-금
    start_time=datetime(2024, 1, 1, 9, 0),   # 9시부터
    end_time=datetime(2024, 12, 31, 18, 0),  # 18시까지
)

# 타임존 지정
trigger = ScheduleTrigger(
    interval_hours=1,
    timezone="America/New_York",
)
```

### Checkpoint에 연결

```python
from truthound.checkpoint import Checkpoint

checkpoint = Checkpoint(
    name="hourly_check",
    data_source="data.csv",
    validators=["null"],
)

# 트리거 추가
checkpoint.add_trigger(ScheduleTrigger(interval_hours=1))
```

---

## CronTrigger

표준 Cron 표현식을 사용하는 트리거입니다.

### 설정 (CronConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `expression` | `str` | `""` | Cron 표현식 (5 또는 6 필드) |
| `timezone` | `str \| None` | `None` | 타임존 |

### Cron 표현식 형식

**5 필드** (표준):
```
분 시 일 월 요일
```

**6 필드** (초 포함):
```
초 분 시 일 월 요일
```

### 필드 값

| 필드 | 범위 | 특수 문자 |
|------|------|-----------|
| 초 (선택) | 0-59 | `*`, `,`, `-`, `/` |
| 분 | 0-59 | `*`, `,`, `-`, `/` |
| 시 | 0-23 | `*`, `,`, `-`, `/` |
| 일 | 1-31 | `*`, `,`, `-`, `/` |
| 월 | 1-12 | `*`, `,`, `-`, `/` |
| 요일 | 0-6 (0=일) | `*`, `,`, `-`, `/` |

### 사용 예시

```python
from truthound.checkpoint.triggers import CronTrigger

# 매일 자정
trigger = CronTrigger(expression="0 0 * * *")

# 매 시간 정각
trigger = CronTrigger(expression="0 * * * *")

# 15분마다
trigger = CronTrigger(expression="*/15 * * * *")

# 매주 월요일 오전 9시
trigger = CronTrigger(expression="0 9 * * 1")

# 평일 오전 9시
trigger = CronTrigger(expression="0 9 * * 1-5")

# 매월 1일 자정
trigger = CronTrigger(expression="0 0 1 * *")

# 6필드: 매일 9시 0분 30초
trigger = CronTrigger(expression="30 0 9 * * *")

# 타임존 지정
trigger = CronTrigger(
    expression="0 9 * * *",
    timezone="Asia/Seoul",
)
```

---

## EventTrigger

외부 이벤트에 의해 트리거됩니다.

### 설정 (EventConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `event_type` | `str` | `""` | 이벤트 타입 (필터링용) |
| `event_filter` | `dict` | `{}` | 이벤트 필터 조건 |
| `debounce_seconds` | `int` | `0` | 디바운스 시간 (초) |
| `batch_events` | `bool` | `False` | 이벤트 배치 처리 |
| `batch_window_seconds` | `int` | `30` | 배치 윈도우 (초) |

### 사용 예시

```python
from truthound.checkpoint.triggers import EventTrigger

# 기본 이벤트 트리거
trigger = EventTrigger(event_type="data_updated")

# 필터 조건 추가
trigger = EventTrigger(
    event_type="data_updated",
    event_filter={
        "source": "production",
        "priority": "high",
    },
)

# 디바운스: 60초 내 중복 이벤트 무시
trigger = EventTrigger(
    event_type="data_updated",
    debounce_seconds=60,
)

# 배치 처리: 30초 동안 이벤트 수집 후 한 번에 처리
trigger = EventTrigger(
    event_type="data_updated",
    batch_events=True,
    batch_window_seconds=30,
)
```

### 이벤트 발생시키기

```python
# 프로그래매틱 이벤트 발생
trigger.fire_event({
    "source": "production",
    "priority": "high",
    "table": "users",
    "rows_affected": 1500,
})
```

---

## FileWatchTrigger

파일 시스템 변경을 감지하여 트리거합니다.

### 설정 (FileWatchConfig)

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `paths` | `list[str]` | `[]` | 감시할 경로 목록 |
| `patterns` | `list[str]` | `["*"]` | 파일 패턴 (glob) |
| `recursive` | `bool` | `True` | 하위 디렉토리 포함 |
| `events` | `list[str]` | `["modified"]` | 감지할 이벤트 |
| `ignore_patterns` | `list[str]` | `[]` | 무시할 패턴 |
| `hash_check` | `bool` | `True` | 해시 기반 변경 감지 |
| `poll_interval_seconds` | `int` | `5` | 폴링 간격 (초) |

### 이벤트 타입

| 이벤트 | 설명 |
|--------|------|
| `modified` | 파일 수정 |
| `created` | 파일 생성 |
| `deleted` | 파일 삭제 |

### 사용 예시

```python
from truthound.checkpoint.triggers import FileWatchTrigger

# 특정 디렉토리 감시
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv", "*.parquet"],
)

# 다중 경로 및 재귀 감시
trigger = FileWatchTrigger(
    paths=["./data", "/shared/datasets"],
    patterns=["*.csv", "*.parquet", "*.json"],
    recursive=True,
    events=["modified", "created"],
)

# 특정 파일 제외
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv"],
    ignore_patterns=[".*", "__pycache__", "*.tmp", "test_*"],
)

# 해시 기반 변경 감지 (실제 내용 변경만 트리거)
trigger = FileWatchTrigger(
    paths=["./data"],
    patterns=["*.csv"],
    hash_check=True,  # 타임스탬프 변경만으로는 트리거 안 됨
    poll_interval_seconds=10,
)
```

---

## CheckpointRunner와 함께 사용

```python
from truthound.checkpoint import Checkpoint, CheckpointRunner
from truthound.checkpoint.triggers import ScheduleTrigger, CronTrigger

# 트리거가 있는 체크포인트 생성
hourly_check = Checkpoint(
    name="hourly_metrics",
    data_source="metrics.csv",
    validators=["null"],
)
hourly_check.add_trigger(ScheduleTrigger(interval_hours=1))

daily_check = Checkpoint(
    name="daily_validation",
    data_source="data.parquet",
    validators=["range", "distribution"],
)
daily_check.add_trigger(CronTrigger(expression="0 0 * * *"))

# Runner 생성 및 실행
runner = CheckpointRunner(
    max_workers=4,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
)

runner.add_checkpoint(hourly_check)
runner.add_checkpoint(daily_check)

# 백그라운드 실행 (트리거 모니터링)
runner.start()

# ... 애플리케이션 로직 ...

# 종료
runner.stop()
```

---

## YAML 설정 예시

```yaml
checkpoints:
  - name: hourly_metrics
    data_source: metrics.csv
    validators:
      - "null"
    triggers:
      # 1시간마다
      - type: schedule
        interval_hours: 1
        run_immediately: true

  - name: daily_validation
    data_source: data.parquet
    validators:
      - range
      - distribution
    triggers:
      # 매일 자정
      - type: cron
        expression: "0 0 * * *"
        timezone: Asia/Seoul

  - name: file_based_check
    data_source: ./data/users.csv
    validators:
      - "null"
    triggers:
      # 파일 변경 시
      - type: file_watch
        paths:
          - ./data
        patterns:
          - "*.csv"
        events:
          - modified
          - created
        hash_check: true
```

---

## 트리거 비교

| 트리거 | 용도 | 장점 | 단점 |
|--------|------|------|------|
| **ScheduleTrigger** | 정기 실행 | 간단, 예측 가능 | 정확한 시간 지정 어려움 |
| **CronTrigger** | 복잡한 스케줄 | 유연한 표현식 | 학습 필요 |
| **EventTrigger** | 이벤트 기반 | 즉시 반응 | 외부 시스템 필요 |
| **FileWatchTrigger** | 파일 변경 | 자동화 | 리소스 소비 |
