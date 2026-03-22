---
title: Logging
---

# Logging

Provides structured logging, sensitive data masking, and platform-specific adapters.

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

## Performance Logging

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

Automatically masks sensitive information:

```python
from common.logging import SensitiveDataMasker

# Automatic masking (password, api_key, token, etc.)
logger.info("Connecting", password="secret")  # password=***MASKED***

# URL credential masking
logger.info("DB: postgres://user:pass@host/db")  # pass -> ***MASKED***
```

## Platform Adapters

Provides logging handlers for each platform:

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

Core logger class:

```python
from common.logging import TruthoundLogger

logger = TruthoundLogger(name="my_module")
logger.info("Message", key="value")
logger.warning("Warning", code=123)
logger.error("Error", exception=e)
```

### Handlers

| Handler | Description |
|---------|-------------|
| `StreamHandler` | Console output |
| `BufferingHandler` | Memory buffering |
| `NullHandler` | No output (for testing) |

### Formatters

| Formatter | Description |
|-----------|-------------|
| `JSONFormatter` | JSON format output |
| `TextFormatter` | Text format output |

### Filters

| Filter | Description |
|--------|-------------|
| `ContextFilter` | Context-based filtering |
| `LevelFilter` | Log level filtering |
| `RegexFilter` | Regex-based filtering |

## LogContext

Nestable context management:

```python
from common.logging import LogContext

with LogContext(request_id="abc123"):
    # All logs within this block include request_id
    with LogContext(user_id="user_1"):
        # Both request_id and user_id included
        logger.info("Processing request")
```

## TimingResult

Performance measurement result:

```python
from common.logging import get_performance_logger

perf = get_performance_logger(__name__)

result = perf.time_sync(expensive_function)
print(f"Duration: {result.duration_ms}ms")
```
