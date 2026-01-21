# Async Execution

비동기 체크포인트 실행을 위한 시스템입니다. 고처리량 워크로드에 적합한 논블로킹 실행을 지원합니다.

## 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                  AsyncCheckpointRunner                          │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Checkpoint 1 │   │ Checkpoint 2 │   │ Checkpoint 3 │        │
│  │  (Triggers)  │   │  (Triggers)  │   │  (Triggers)  │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Async Execution Pool (Semaphore)            │  │
│  │                                                          │  │
│  │   Task 1 ──┐                                             │  │
│  │   Task 2 ──┼── await gather(*) ──▶ Results Queue         │  │
│  │   Task 3 ──┘                                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 클래스

### AsyncBaseAction

비동기 액션 기본 클래스입니다.

```python
from truthound.checkpoint.async_base import AsyncBaseAction, ActionConfig

class MyAsyncAction(AsyncBaseAction[MyConfig]):
    action_type = "my_async_action"

    @classmethod
    def _default_config(cls) -> MyConfig:
        return MyConfig()

    async def _execute_async(self, checkpoint_result) -> ActionResult:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as resp:
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    action_name=self.name,
                    action_type=self.action_type,
                )
```

**주요 기능:**
- 자동 타임아웃 (`asyncio.wait_for`)
- 지수 백오프 재시도
- 취소 지원 (`CancelledError`)
- `notify_on` 조건 확인

### AsyncBaseTrigger

비동기 트리거 기본 클래스입니다.

```python
from truthound.checkpoint.async_base import AsyncBaseTrigger, TriggerConfig

class MyAsyncTrigger(AsyncBaseTrigger[MyTriggerConfig]):
    trigger_type = "my_async_trigger"

    @classmethod
    def _default_config(cls) -> MyTriggerConfig:
        return MyTriggerConfig()

    async def should_trigger_async(self) -> TriggerResult:
        # 비동기 조건 확인
        message = await self._consumer.poll()
        return TriggerResult(should_run=message is not None)
```

---

## 프로토콜

### AsyncExecutable

```python
from truthound.checkpoint.async_base import AsyncExecutable

@runtime_checkable
class AsyncExecutable(Protocol):
    async def execute_async(self, checkpoint_result) -> ActionResult:
        ...
```

### SyncExecutable

```python
from truthound.checkpoint.async_base import SyncExecutable

@runtime_checkable
class SyncExecutable(Protocol):
    def execute(self, checkpoint_result) -> ActionResult:
        ...
```

---

## SyncActionAdapter

동기 액션을 비동기 컨텍스트에서 실행합니다.

```python
from truthound.checkpoint.async_base import SyncActionAdapter, adapt_to_async
from truthound.checkpoint.actions import SlackNotification

# 동기 액션 래핑
sync_action = SlackNotification(webhook_url="...")
async_action = SyncActionAdapter(sync_action)

# 또는 adapt_to_async 사용
async_action = adapt_to_async(sync_action)

# 비동기로 실행 (ThreadPoolExecutor 사용)
await async_action.execute_async(checkpoint_result)
```

### adapt_to_async 함수

```python
from concurrent.futures import ThreadPoolExecutor

# 커스텀 Executor 사용
executor = ThreadPoolExecutor(max_workers=8)
async_action = adapt_to_async(sync_action, executor=executor)
```

---

## 실행 전략 (3가지)

### SequentialStrategy

순차적으로 액션을 실행합니다.

```python
from truthound.checkpoint.async_base import (
    SequentialStrategy,
    AsyncExecutionContext,
)

strategy = SequentialStrategy()
context = AsyncExecutionContext(cancel_on_first_error=True)

results = await strategy.execute(
    actions=[action1, action2, action3],
    checkpoint_result=result,
    context=context,
)
```

**특징:**
- 순서 보장
- 오류 시 중단 가능 (`cancel_on_first_error`)
- 의존성 있는 액션에 적합

### ConcurrentStrategy

모든 액션을 동시에 실행합니다.

```python
from truthound.checkpoint.async_base import ConcurrentStrategy

strategy = ConcurrentStrategy(max_concurrency=5)

results = await strategy.execute(
    actions=actions,
    checkpoint_result=result,
)
```

**특징:**
- 병렬 실행
- 동시성 제한 가능 (`max_concurrency`)
- 독립 액션에 적합

### PipelineStrategy

파이프라인 스테이지로 실행합니다.

```python
from truthound.checkpoint.async_base import PipelineStrategy

# 3개 스테이지: [0,1] → [2] → [3,4]
strategy = PipelineStrategy(stages=[[0, 1], [2], [3, 4]])

results = await strategy.execute(
    actions=[a0, a1, a2, a3, a4],
    checkpoint_result=result,
)
```

**특징:**
- 스테이지 내 병렬 실행
- 스테이지 간 순차 실행
- 복잡한 의존성 그래프에 적합

---

## AsyncExecutionContext

비동기 실행 컨텍스트입니다.

```python
from truthound.checkpoint.async_base import AsyncExecutionContext

context = AsyncExecutionContext(
    executor=ThreadPoolExecutor(max_workers=4),  # 동기 액션용
    semaphore=asyncio.Semaphore(10),             # 동시성 제한
    timeout=30.0,                                 # 기본 타임아웃
    cancel_on_first_error=False,                  # 오류 시 중단
)

async with context:
    results = await strategy.execute(actions, result, context)
```

---

## AsyncCheckpointRunner

비동기 체크포인트 러너입니다.

```python
from truthound.checkpoint.async_runner import (
    AsyncCheckpointRunner,
    AsyncRunnerConfig,
)

# 설정
config = AsyncRunnerConfig(
    max_concurrent_checkpoints=10,    # 동시 체크포인트 수
    trigger_poll_interval=1.0,        # 트리거 폴링 간격
    result_queue_size=1000,           # 결과 큐 크기
    stop_on_error=False,              # 오류 시 중단
    max_consecutive_failures=10,      # 최대 연속 실패
    graceful_shutdown_timeout=30.0,   # 셧다운 타임아웃
)

# 러너 생성
runner = AsyncCheckpointRunner(
    config=config,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
    error_callback=lambda e: print(f"Error: {e}"),
)

# 체크포인트 등록
runner.add_checkpoint(checkpoint1)
runner.add_checkpoint(checkpoint2)

# 백그라운드 실행 시작
await runner.start_async()

# 결과 스트리밍
async for result in runner.iter_results_async():
    print(f"Result: {result.status}")

# 종료
await runner.stop_async()
```

### 단일 실행

```python
# 이름으로 실행
result = await runner.run_once_async("my_checkpoint")

# 객체로 실행
result = await runner.run_once_async(checkpoint, context={"key": "value"})
```

### 전체 실행

```python
results = await runner.run_all_async(context={"shared": "data"})
```

### 통계 조회

```python
stats = runner.get_stats()
print(f"Running: {stats['running']}")
print(f"Checkpoints: {stats['checkpoints']}")
print(f"Pending tasks: {stats['pending_tasks']}")
print(f"Queued results: {stats['queued_results']}")
```

---

## 편의 함수

### run_checkpoint_async

단일 체크포인트를 비동기로 실행합니다.

```python
from truthound.checkpoint.async_runner import run_checkpoint_async

# 이름으로 실행 (레지스트리에서 조회)
result = await run_checkpoint_async("my_checkpoint")

# 객체로 실행
result = await run_checkpoint_async(checkpoint, context={"key": "value"})
```

### run_checkpoints_parallel

여러 체크포인트를 병렬로 실행합니다.

```python
from truthound.checkpoint.async_runner import run_checkpoints_parallel

results = await run_checkpoints_parallel(
    checkpoints=[cp1, cp2, cp3],
    max_concurrent=5,
    context={"shared": "data"},
    on_complete=lambda r: print(f"Done: {r.checkpoint_name}"),
)
```

---

## CheckpointPool

고처리량 시나리오를 위한 워커 풀입니다.

```python
from truthound.checkpoint.async_runner import CheckpointPool

async with CheckpointPool(workers=10) as pool:
    # 단일 제출
    result = await pool.submit(checkpoint)

    # 다중 제출
    results = await pool.submit_many([cp1, cp2, cp3])
```

### 수동 관리

```python
pool = CheckpointPool(
    workers=10,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
)

await pool.start()

# 작업 제출...
result = await pool.submit(checkpoint)

await pool.stop()
```

---

## 데코레이터

### @with_retry

재시도 로직을 추가합니다.

```python
from truthound.checkpoint.async_base import with_retry

@with_retry(max_retries=3, delay=1.0, backoff=2.0)
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
```

### @with_timeout

타임아웃을 추가합니다.

```python
from truthound.checkpoint.async_base import with_timeout

@with_timeout(seconds=30.0)
async def slow_operation():
    await asyncio.sleep(60)  # TimeoutError 발생
```

### @with_semaphore

동시성을 제한합니다.

```python
from truthound.checkpoint.async_base import with_semaphore

semaphore = asyncio.Semaphore(5)

@with_semaphore(semaphore)
async def limited_operation():
    # 최대 5개 동시 실행
    await do_work()
```

---

## 전체 예시

```python
import asyncio
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.async_runner import (
    AsyncCheckpointRunner,
    AsyncRunnerConfig,
    run_checkpoints_parallel,
)
from truthound.checkpoint.async_base import (
    AsyncBaseAction,
    ConcurrentStrategy,
    adapt_to_async,
)
from truthound.checkpoint.actions import SlackNotification
from truthound.checkpoint.triggers import ScheduleTrigger


# 커스텀 비동기 액션
class AsyncWebhookAction(AsyncBaseAction):
    action_type = "async_webhook"

    async def _execute_async(self, result):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._config.url,
                json=result.to_dict(),
            ) as resp:
                return ActionResult(
                    status=ActionStatus.SUCCESS if resp.ok else ActionStatus.ERROR,
                    action_name=self.name,
                    action_type=self.action_type,
                )


# 동기 액션을 비동기로 변환
slack = SlackNotification(webhook_url="${SLACK_WEBHOOK}")
async_slack = adapt_to_async(slack)

# 체크포인트 생성
checkpoint1 = Checkpoint(
    name="hourly_validation",
    data_source="data.csv",
    validators=["null"],
    actions=[async_slack],
)
checkpoint1.add_trigger(ScheduleTrigger(interval_hours=1))

checkpoint2 = Checkpoint(
    name="daily_validation",
    data_source="data.parquet",
    validators=["range", "distribution"],
)


async def main():
    # 1. 병렬 실행
    results = await run_checkpoints_parallel(
        checkpoints=[checkpoint1, checkpoint2],
        max_concurrent=5,
        on_complete=lambda r: print(f"Done: {r.checkpoint_name}"),
    )

    # 2. 러너로 트리거 기반 실행
    runner = AsyncCheckpointRunner(
        config=AsyncRunnerConfig(max_concurrent_checkpoints=10),
        result_callback=lambda r: print(f"Completed: {r.status}"),
    )

    runner.add_checkpoint(checkpoint1)
    runner.add_checkpoint(checkpoint2)

    await runner.start_async()

    # 결과 스트리밍
    async for result in runner.iter_results_async(timeout=5.0):
        print(f"{result.checkpoint_name}: {result.status}")

        # 10개 결과 후 종료
        if runner._result_queue.qsize() >= 10:
            break

    await runner.stop_async()


asyncio.run(main())
```

---

## 동기/비동기 혼합 사용

동기 액션과 비동기 액션을 같은 체크포인트에서 사용할 수 있습니다.

```python
from truthound.checkpoint.async_base import adapt_to_async, ConcurrentStrategy

# 동기 액션
sync_action = SlackNotification(webhook_url="...")

# 비동기 액션
async_action = MyAsyncWebhook(url="...")

# 모두 비동기로 변환
actions = [
    adapt_to_async(sync_action),  # ThreadPoolExecutor에서 실행
    async_action,                  # 네이티브 비동기 실행
]

# 동시 실행
strategy = ConcurrentStrategy(max_concurrency=5)
results = await strategy.execute(actions, checkpoint_result)
```

---

## 성능 고려사항

| 시나리오 | 권장 설정 |
|----------|-----------|
| I/O 바운드 | `ConcurrentStrategy` + 높은 `max_concurrency` |
| CPU 바운드 | `SequentialStrategy` 또는 낮은 `max_concurrency` |
| 의존성 있음 | `PipelineStrategy` 또는 `SequentialStrategy` |
| 고처리량 | `CheckpointPool` + `workers=CPU_COUNT * 2` |
| 메모리 제한 | `result_queue_size` 제한 + 백프레셔 처리 |
