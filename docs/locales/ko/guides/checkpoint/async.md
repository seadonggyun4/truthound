# Async Execution

실무 운영 가이드에서 Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Core Classes

### AsyncBaseAction

실무 운영 가이드에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Key을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `asyncio.wait_for`, Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Exponential backoff 재시도
- 실무 운영 가이드에서 `CancelledError`, Cancellation, CancelledError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `notify_on`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### AsyncBaseTrigger

실무 운영 가이드에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import AsyncBaseTrigger, TriggerConfig

class MyAsyncTrigger(AsyncBaseTrigger[MyTriggerConfig]):
    trigger_type = "my_async_trigger"

    @classmethod
    def _default_config(cls) -> MyTriggerConfig:
        return MyTriggerConfig()

    async def should_trigger_async(self) -> TriggerResult:
        # Asynchronous condition check
        message = await self._consumer.poll()
        return TriggerResult(should_run=message is not None)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Protocols

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SyncActionAdapter

실무 운영 가이드에서 Executes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import SyncActionAdapter, adapt_to_async
from truthound.checkpoint.actions import SlackNotification

# Wrap synchronous action
sync_action = SlackNotification(webhook_url="...")
async_action = SyncActionAdapter(sync_action)

# Or use adapt_to_async
async_action = adapt_to_async(sync_action)

# Execute asynchronously (uses ThreadPoolExecutor)
await async_action.execute_async(checkpoint_result)
```

### adapt_to_async Function

```python
from concurrent.futures import ThreadPoolExecutor

# Use custom executor
executor = ThreadPoolExecutor(max_workers=8)
async_action = adapt_to_async(sync_action, executor=executor)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Execution Strategies (3 Types)

### SequentialStrategy

실무 운영 가이드에서 Executes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Characteristics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Order을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `cancel_on_first_error`, Can을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Suitable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### ConcurrentStrategy

실무 운영 가이드에서 Executes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import ConcurrentStrategy

strategy = ConcurrentStrategy(max_concurrency=5)

results = await strategy.execute(
    actions=actions,
    checkpoint_result=result,
)
```

실무 운영 가이드에서 Characteristics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `max_concurrency`, Concurrency을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Suitable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### PipelineStrategy

Executes actions in 파이프라인 stages.

```python
from truthound.checkpoint.async_base import PipelineStrategy

# 3 stages: [0,1] → [2] → [3,4]
strategy = PipelineStrategy(stages=[[0, 1], [2], [3, 4]])

results = await strategy.execute(
    actions=[a0, a1, a2, a3, a4],
    checkpoint_result=result,
)
```

실무 운영 가이드에서 Characteristics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Sequential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Suitable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## AsyncExecutionContext

실무 운영 가이드에서 Asynchronous을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import AsyncExecutionContext

context = AsyncExecutionContext(
    executor=ThreadPoolExecutor(max_workers=4),  # For sync actions
    semaphore=asyncio.Semaphore(10),             # Concurrency limit
    timeout=30.0,                                 # Default timeout
    cancel_on_first_error=False,                  # Abort on error
)

async with context:
    results = await strategy.execute(actions, result, context)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Async체크포인트Runner

Asynchronous 체크포인트 runner.

```python
from truthound.checkpoint.async_runner import (
    AsyncCheckpointRunner,
    AsyncRunnerConfig,
)

# Configuration
config = AsyncRunnerConfig(
    max_concurrent_checkpoints=10,    # Concurrent checkpoints
    trigger_poll_interval=1.0,        # Trigger polling interval
    result_queue_size=1000,           # Result queue size
    stop_on_error=False,              # Stop on error
    max_consecutive_failures=10,      # Maximum consecutive failures
    graceful_shutdown_timeout=30.0,   # Shutdown timeout
)

# Create runner
runner = AsyncCheckpointRunner(
    config=config,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
    error_callback=lambda e: print(f"Error: {e}"),
)

# Register checkpoints
runner.add_checkpoint(checkpoint1)
runner.add_checkpoint(checkpoint2)

# Start background execution
await runner.start_async()

# Stream results
async for result in runner.iter_results_async():
    print(f"Result: {result.status}")

# Shutdown
await runner.stop_async()
```

### Single Execution

```python
# Execute by name
result = await runner.run_once_async("my_checkpoint")

# Execute by object
result = await runner.run_once_async(checkpoint, context={"key": "value"})
```

### Execute All

```python
results = await runner.run_all_async(context={"shared": "data"})
```

### Query Statistics

```python
stats = runner.get_stats()
print(f"Running: {stats['running']}")
print(f"Checkpoints: {stats['checkpoints']}")
print(f"Pending tasks: {stats['pending_tasks']}")
print(f"Queued results: {stats['queued_results']}")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Convenience Functions

### run_checkpoint_async

Executes a single 체크포인트 asynchronously.

```python
from truthound.checkpoint.async_runner import run_checkpoint_async

# Execute by name (looks up in registry)
result = await run_checkpoint_async("my_checkpoint")

# Execute by object
result = await run_checkpoint_async(checkpoint, context={"key": "value"})
```

### run_checkpoints_parallel

Executes multiple 체크포인트 in parallel.

```python
from truthound.checkpoint.async_runner import run_checkpoints_parallel

results = await run_checkpoints_parallel(
    checkpoints=[cp1, cp2, cp3],
    max_concurrent=5,
    context={"shared": "data"},
    on_complete=lambda r: print(f"Done: {r.checkpoint_name}"),
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 체크포인트Pool

실무 운영 가이드에서 Worker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_runner import CheckpointPool

async with CheckpointPool(workers=10) as pool:
    # Single submission
    result = await pool.submit(checkpoint)

    # Multiple submissions
    results = await pool.submit_many([cp1, cp2, cp3])
```

### Manual Management

```python
pool = CheckpointPool(
    workers=10,
    result_callback=lambda r: print(f"Completed: {r.checkpoint_name}"),
)

await pool.start()

# Submit work...
result = await pool.submit(checkpoint)

await pool.stop()
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Decorators

### @with_retry

Adds 재시도 logic.

```python
from truthound.checkpoint.async_base import with_retry

@with_retry(max_retries=3, delay=1.0, backoff=2.0)
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
```

### @with_timeout

실무 운영 가이드에서 Adds을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import with_timeout

@with_timeout(seconds=30.0)
async def slow_operation():
    await asyncio.sleep(60)  # Raises TimeoutError
```

### @with_semaphore

실무 운영 가이드에서 Limits을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import with_semaphore

semaphore = asyncio.Semaphore(5)

@with_semaphore(semaphore)
async def limited_operation():
    # Maximum 5 concurrent executions
    await do_work()
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Complete Example

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


# Custom async action
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


# Convert synchronous action to async
slack = SlackNotification(webhook_url="${SLACK_WEBHOOK}")
async_slack = adapt_to_async(slack)

# Create checkpoints
checkpoint1 = Checkpoint(
    name="hourly_validation",
    data_source="data.csv",
    validators=["null"],
    actions=[async_slack],
)
checkpoint1.add_trigger(ScheduleTrigger(interval_hours=1))

checkpoint2 = Checkpoint(
    name="daily_data_validation",
    data_source="data.parquet",
    validators=["range", "distribution"],
)


async def main():
    # 1. Parallel execution
    results = await run_checkpoints_parallel(
        checkpoints=[checkpoint1, checkpoint2],
        max_concurrent=5,
        on_complete=lambda r: print(f"Done: {r.checkpoint_name}"),
    )

    # 2. Trigger-based execution with runner
    runner = AsyncCheckpointRunner(
        config=AsyncRunnerConfig(max_concurrent_checkpoints=10),
        result_callback=lambda r: print(f"Completed: {r.status}"),
    )

    runner.add_checkpoint(checkpoint1)
    runner.add_checkpoint(checkpoint2)

    await runner.start_async()

    # Stream results
    async for result in runner.iter_results_async(timeout=5.0):
        print(f"{result.checkpoint_name}: {result.status}")

        # Stop after 10 results
        if runner._result_queue.qsize() >= 10:
            break

    await runner.stop_async()


asyncio.run(main())
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Mixed Sync/Async Usage

실무 운영 가이드에서 Synchronous을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.checkpoint.async_base import adapt_to_async, ConcurrentStrategy

# Synchronous action
sync_action = SlackNotification(webhook_url="...")

# Asynchronous action
async_action = MyAsyncWebhook(url="...")

# Convert all to async
actions = [
    adapt_to_async(sync_action),  # Runs in ThreadPoolExecutor
    async_action,                  # Native async execution
]

# Execute concurrently
strategy = ConcurrentStrategy(max_concurrency=5)
results = await strategy.execute(actions, checkpoint_result)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 성능 Considerations

| 실무 운영 가이드에서 Scenario을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Recommended 설정 |
|----------|--------------------------|
| 실무 운영 가이드에서 I/O을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `ConcurrentStrategy`, `max_concurrency`, ConcurrentStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 CPU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `SequentialStrategy`, `max_concurrency`, SequentialStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Has을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `PipelineStrategy`, `SequentialStrategy`, PipelineStrategy, SequentialStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 High을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `CheckpointPool`, `workers=CPU_COUNT * 2`, CheckpointPool, CPU_COUNT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `result_queue_size`, Limit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
