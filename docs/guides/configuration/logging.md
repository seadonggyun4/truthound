# Logging Configuration

Truthound provides an enterprise-grade structured logging system with correlation IDs, multiple output sinks, and async buffered logging.

## Architecture

```
CorrelationContext (thread-local)
       |
       v
EnterpriseLogger
       |
       +---> LogSink[] (parallel dispatch)
               |
               +---> ConsoleSink
               +---> FileSink
               +---> JsonFileSink
               +---> ElasticsearchSink
               +---> LokiSink
               +---> FluentdSink
```

## Quick Start

```python
from truthound.infrastructure.logging import (
    configure_logging,
    get_logger,
    correlation_context,
)

# Configure for production
configure_logging(
    level="info",
    format="json",
    service="truthound",
    environment="production",
    sinks=[
        {"type": "console"},
        {"type": "elasticsearch", "url": "http://elk:9200"},
    ],
)

# Get logger
logger = get_logger(__name__)
logger.info("Application started", version="1.0.0")

# Use correlation context
with correlation_context(request_id="req-123", user_id="user-456"):
    logger.info("Processing request")  # Includes request_id and user_id
```

## Log Levels

| Level | Value | Description |
|-------|-------|-------------|
| `TRACE` | 5 | Detailed debugging information |
| `DEBUG` | 10 | Debug information |
| `INFO` | 20 | Informational messages |
| `WARNING` | 30 | Warning messages |
| `ERROR` | 40 | Error messages |
| `CRITICAL` | 50 | Critical errors |
| `AUDIT` | 60 | Special level for audit events |

```python
from truthound.infrastructure.logging import LogLevel

# Convert from string
level = LogLevel.from_string("info")  # LogLevel.INFO
level = LogLevel.from_string("warn")  # LogLevel.WARNING
level = LogLevel.from_string("fatal") # LogLevel.CRITICAL
```

## LogConfig

Configuration dataclass for logging.

```python
from truthound.infrastructure.logging import LogConfig, LogLevel

config = LogConfig(
    level=LogLevel.INFO,         # Log level (str or LogLevel)
    format="json",               # Output format: console, json, logfmt
    service="truthound",         # Service name for identification
    environment="production",    # Environment name
    include_caller=False,        # Include file:line:function
    include_meta=True,           # Include thread/process info
    sinks=[{"type": "console"}], # Sink configurations
    async_logging=True,          # Enable async buffered logging
    buffer_size=1000,            # Buffer size for async logging
    flush_interval=1.0,          # Flush interval in seconds
)
```

### Presets

```python
# Development preset
# level=DEBUG, format=console, include_caller=True, async_logging=False
config = LogConfig.development()

# Production preset
# level=INFO, format=json, async_logging=True
config = LogConfig.production("my-service")

# From environment variables
# Uses LOG_LEVEL, LOG_FORMAT, SERVICE_NAME, ENVIRONMENT
config = LogConfig.from_environment()
```

### Environment Variables

| Variable | LogConfig Field | Default |
|----------|-----------------|---------|
| `LOG_LEVEL` | `level` | `INFO` |
| `LOG_FORMAT` | `format` | `console` |
| `SERVICE_NAME` | `service` | `""` |
| `ENVIRONMENT` | `environment` | `development` |
| `LOG_INCLUDE_CALLER` | `include_caller` | `false` |

## Log Sinks

### ConsoleSink

Outputs logs to stdout/stderr with optional coloring.

```python
from truthound.infrastructure.logging import ConsoleSink, LogLevel

sink = ConsoleSink(
    stream=None,                 # Output stream (None for auto)
    color=True,                  # Enable ANSI colors
    format="console",            # console, json, logfmt
    split_stderr=True,           # Send warnings+ to stderr
    timestamp_format="%Y-%m-%d %H:%M:%S",
    level=LogLevel.DEBUG,        # Minimum level to accept
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "console",
        "color": True,
        "format": "json",
    }
]
```

### FileSink

File output with rotation support.

```python
from truthound.infrastructure.logging import FileSink

sink = FileSink(
    path="/var/log/truthound.log",
    format="json",               # json, logfmt, text
    max_bytes=10 * 1024 * 1024,  # 10MB before rotation
    backup_count=5,              # Number of backup files
    encoding="utf-8",
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "file",
        "path": "/var/log/truthound.log",
        "format": "json",
        "max_bytes": 10485760,
        "backup_count": 5,
    }
]
```

### JsonFileSink

Convenience wrapper for JSON file output.

```python
from truthound.infrastructure.logging import JsonFileSink

sink = JsonFileSink(
    path="/var/log/truthound.jsonl",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "json_file",
        "path": "/var/log/truthound.jsonl",
    }
]
```

### ElasticsearchSink

Sends logs to Elasticsearch/OpenSearch with bulk indexing.

```python
from truthound.infrastructure.logging import ElasticsearchSink

sink = ElasticsearchSink(
    url="http://elasticsearch:9200",
    index_prefix="truthound-logs",
    index_pattern="daily",       # daily, weekly, monthly
    username=None,               # Basic auth username
    password=None,               # Basic auth password
    api_key=None,                # API key for auth
    batch_size=100,              # Records per bulk request
    flush_interval=5.0,          # Flush interval in seconds
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "elasticsearch",
        "url": "https://es.example.com:9200",
        "index_prefix": "truthound-logs",
        "index_pattern": "daily",
        "api_key": "${ES_API_KEY}",
    }
]
```

Index naming based on pattern:

| Pattern | Index Name Example |
|---------|-------------------|
| `daily` | `truthound-logs-2024.01.15` |
| `weekly` | `truthound-logs-2024.03` |
| `monthly` | `truthound-logs-2024.01` |

### LokiSink

Sends logs to Grafana Loki with labels.

```python
from truthound.infrastructure.logging import LokiSink

sink = LokiSink(
    url="http://loki:3100/loki/api/v1/push",
    labels={
        "app": "truthound",
        "env": "production",
    },
    batch_size=100,
    flush_interval=5.0,
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "loki",
        "url": "http://loki:3100/loki/api/v1/push",
        "labels": {"app": "truthound", "env": "production"},
    }
]
```

### FluentdSink

Sends logs to Fluentd using the Forward protocol.

```python
from truthound.infrastructure.logging import FluentdSink

sink = FluentdSink(
    host="localhost",
    port=24224,
    tag="truthound",             # Tag prefix
)
```

Configuration via dict:

```python
sinks = [
    {
        "type": "fluentd",
        "host": "fluentd.example.com",
        "port": 24224,
        "tag": "truthound.logs",
    }
]
```

## Correlation Context

Thread-local context for distributed tracing.

### Basic Usage

```python
from truthound.infrastructure.logging import (
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
)

# Auto-generate correlation ID
with correlation_context():
    logger.info("Processing")  # Includes auto-generated correlation_id

# With explicit fields
with correlation_context(
    request_id="req-123",
    user_id="user-456",
    trace_id="trace-789",
):
    logger.info("User action")  # Includes all context fields
    call_downstream_service()   # Context propagates

# Manual management
set_correlation_id("custom-id-456")
current_id = get_correlation_id()
```

### HTTP Header Propagation

```python
from truthound.infrastructure.logging import CorrelationContext

# Convert context to HTTP headers for outgoing requests
headers = CorrelationContext.to_headers()
# {'X-Correlation-Request-Id': 'req-123', 'X-Correlation-User-Id': 'user-456'}

# Extract context from incoming HTTP headers
context = CorrelationContext.from_headers(request.headers)
with correlation_context(**context):
    process_request()
```

### Nested Contexts

```python
with correlation_context(request_id="req-123"):
    logger.info("Outer context")  # request_id=req-123

    with correlation_context(span_id="span-456"):
        logger.info("Inner context")  # request_id=req-123, span_id=span-456

    logger.info("Back to outer")  # request_id=req-123
```

## EnterpriseLogger

Full-featured logger with field binding and correlation support.

### Basic Usage

```python
from truthound.infrastructure.logging import EnterpriseLogger, LogConfig

logger = EnterpriseLogger(
    "my.module",
    config=LogConfig.production("my-service"),
)

logger.info("Request received", path="/api/users", method="GET")
logger.warning("Rate limit approaching", current=95, limit=100)
logger.error("Database connection failed", host="db.example.com")
```

### Field Binding

Create child loggers with bound fields:

```python
# Create logger with bound fields
request_logger = logger.bind(
    request_id="abc123",
    user_id="user-456",
    ip_address="192.168.1.1",
)

# All logs include bound fields automatically
request_logger.info("Authentication successful")
request_logger.info("Processing order", order_id="ord-789")
```

### Exception Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.exception("Operation failed", operation="risky")
    # Automatically includes exception type, message, and traceback
```

### Logging Methods

| Method | Level | Description |
|--------|-------|-------------|
| `trace(msg, **fields)` | TRACE | Detailed debugging |
| `debug(msg, **fields)` | DEBUG | Debug information |
| `info(msg, **fields)` | INFO | Informational messages |
| `warning(msg, **fields)` | WARNING | Warning messages |
| `warn(msg, **fields)` | WARNING | Alias for warning |
| `error(msg, **fields)` | ERROR | Error messages |
| `critical(msg, **fields)` | CRITICAL | Critical errors |
| `fatal(msg, **fields)` | CRITICAL | Alias for critical |
| `exception(msg, exc, **fields)` | ERROR | Log with exception |
| `audit(msg, **fields)` | AUDIT | Audit events |

## Output Formats

### Console Format

Human-readable format for development:

```
2024-01-15 10:30:45 INFO     [abc12345] [my.module] Request received path=/api/users method=GET
```

### JSON Format

Structured format for log aggregation:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456+00:00",
  "@timestamp": "2024-01-15T10:30:45.123456+00:00",
  "level": "info",
  "message": "Request received",
  "logger": "my.module",
  "correlation_id": "abc12345",
  "service": "truthound",
  "environment": "production",
  "path": "/api/users",
  "method": "GET",
  "thread_id": 12345,
  "thread_name": "MainThread",
  "process_id": 6789,
  "hostname": "web-1"
}
```

### Logfmt Format

Key-value format for structured logging:

```
ts=2024-01-15T10:30:45.123456+00:00 level=info msg="Request received" logger=my.module correlation_id=abc12345 path=/api/users method=GET
```

## Global Configuration

### configure_logging

```python
from truthound.infrastructure.logging import configure_logging

configure_logging(
    level="info",
    format="json",
    service="truthound",
    environment="production",
    sinks=[
        {"type": "console"},
        {"type": "file", "path": "/var/log/truthound.log"},
        {"type": "elasticsearch", "url": "http://elk:9200"},
    ],
    async_logging=True,
    buffer_size=1000,
    include_caller=False,
)
```

### get_logger

```python
from truthound.infrastructure.logging import get_logger

# Get or create logger (uses global config)
logger = get_logger(__name__)
logger = get_logger("my.module.name")
```

### reset_logging

```python
from truthound.infrastructure.logging import reset_logging

# Reset to defaults (closes all loggers)
reset_logging()
```

## Production Example

```python
from truthound.infrastructure.logging import (
    configure_logging,
    get_logger,
    correlation_context,
)

# Configure for production
configure_logging(
    level="info",
    format="json",
    service="truthound-api",
    environment="production",
    sinks=[
        # Console for container logs
        {"type": "console", "format": "json"},
        # File backup
        {
            "type": "file",
            "path": "/var/log/truthound/api.log",
            "max_bytes": 52428800,  # 50MB
            "backup_count": 10,
        },
        # Elasticsearch for centralized logging
        {
            "type": "elasticsearch",
            "url": "https://es.example.com:9200",
            "index_prefix": "truthound-api",
            "api_key": "${ES_API_KEY}",
        },
        # Loki for Grafana
        {
            "type": "loki",
            "url": "http://loki:3100/loki/api/v1/push",
            "labels": {"app": "truthound-api", "env": "production"},
        },
    ],
    async_logging=True,
    buffer_size=2000,
)

# Application code
logger = get_logger(__name__)

def handle_request(request):
    with correlation_context(
        request_id=request.headers.get("X-Request-ID"),
        user_id=request.user_id,
    ):
        logger.info("Request started", path=request.path, method=request.method)
        try:
            result = process_request(request)
            logger.info("Request completed", status=200)
            return result
        except Exception as e:
            logger.exception("Request failed", status=500)
            raise
```
