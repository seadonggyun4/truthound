---
title: 로깅
---

# 로깅

오케스트레이션 실행에서 Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from common.logging import get_logger, LogContext

logger = get_logger(__name__)

# Structured logging
logger.info("Processing data", rows=1000, platform="airflow")

# Context propagation
with LogContext(operation="validate", task_id="task_1"):
    logger.info("Starting validation")  # operation, task_id automatically included
    with LogContext(column="email"):
        logger.warning("Null values found")  # All context included
```

## 성능 로깅

```python
from common.logging import get_performance_logger

perf = get_performance_logger(__name__)

with perf.timed("database_query", table="users"):
    result = execute_query()
# Automatic logging: "database_query completed in 123.45ms"

@perf.timed_decorator()
def process_batch(data):
    return transform(data)
```

## Sensitive Data Masking

오케스트레이션 실행에서 Automatically을(를) 다루는 항목입니다:

```python
from common.logging import SensitiveDataMasker

# Automatic masking (password, api_key, token, etc.)
logger.info("Connecting", password="secret")  # password=***MASKED***

# URL credential masking
logger.info("DB: postgres://user:pass@host/db")  # pass -> ***MASKED***
```

## 플랫폼 Adapters

오케스트레이션 실행에서 Provides을(를) 다루는 항목입니다:

```python
from common.logging import AirflowLoggerAdapter, DagsterLoggerAdapter, PrefectLoggerAdapter

# Airflow task logging
adapter = AirflowLoggerAdapter(task_instance)
logger.add_handler(adapter)

# Dagster op logging
adapter = DagsterLoggerAdapter(context)

# Prefect flow logging
adapter = PrefectLoggerAdapter()
```

## Logger Components

### TruthoundLogger

오케스트레이션 실행에서 Core을(를) 다루는 항목입니다:

```python
from common.logging import TruthoundLogger

logger = TruthoundLogger(name="my_module")
logger.info("Message", key="value")
logger.warning("Warning", code=123)
logger.error("Error", exception=e)
```

### Handlers

| 오케스트레이션 실행에서 Handler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|
| 오케스트레이션 실행에서 `StreamHandler`, StreamHandler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Console을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `BufferingHandler`, BufferingHandler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `NullHandler`, NullHandler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Formatters

| 오케스트레이션 실행에서 Formatter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| 오케스트레이션 실행에서 JSON, `JSONFormatter`, JSONFormatter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `TextFormatter`, TextFormatter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Text을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Filters

| 오케스트레이션 실행에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 오케스트레이션 실행에서 `ContextFilter`, ContextFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Context-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LevelFilter`, LevelFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Log을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `RegexFilter`, RegexFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Regex-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## LogContext

오케스트레이션 실행에서 Nestable을(를) 다루는 항목입니다:

```python
from common.logging import LogContext

with LogContext(request_id="abc123"):
    # All logs within this block include request_id
    with LogContext(user_id="user_1"):
        # Both request_id and user_id included
        logger.info("Processing request")
```

## TimingResult

성능 measurement 결과:

```python
from common.logging import get_performance_logger

perf = get_performance_logger(__name__)

result = perf.time_sync(expensive_function)
print(f"Duration: {result.duration_ms}ms")
```
