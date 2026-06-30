# Store 관측성

실무 운영 가이드에서 Comprehensive, Prometheus, OpenTelemetry을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Audit, Logging, Track을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Metrics, Prometheus-compatible을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Tracing, Distributed, OpenTelemetry을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 감사 로깅

### 감사 Event Types

```python
from truthound.stores.observability.audit import AuditEventType

# Data operations
AuditEventType.CREATE          # Item created
AuditEventType.READ            # Item read
AuditEventType.UPDATE          # Item updated
AuditEventType.DELETE          # Item deleted
AuditEventType.QUERY           # Query executed
AuditEventType.LIST            # Items listed
AuditEventType.COUNT           # Count operation

# Lifecycle events
AuditEventType.INITIALIZE      # Store initialized
AuditEventType.CLOSE           # Store closed
AuditEventType.FLUSH           # Data flushed

# Batch operations
AuditEventType.BATCH_CREATE    # Batch create
AuditEventType.BATCH_DELETE    # Batch delete

# Replication & sync
AuditEventType.REPLICATE       # Data replicated
AuditEventType.SYNC            # Sync operation
AuditEventType.MIGRATE         # Data migrated
AuditEventType.ROLLBACK        # Rollback operation

# Access control
AuditEventType.ACCESS_DENIED   # Access denied
AuditEventType.ACCESS_GRANTED  # Access granted

# Errors
AuditEventType.ERROR           # General error
AuditEventType.VALIDATION_ERROR # Validation error
```

### 감사 Status

```python
from truthound.stores.observability.audit import AuditStatus

AuditStatus.SUCCESS    # Operation succeeded
AuditStatus.FAILURE    # Operation failed
AuditStatus.PARTIAL    # Partial success
AuditStatus.DENIED     # Access denied
```

### 감사Event

```python
from truthound.stores.observability.audit import AuditEvent, AuditEventType, AuditStatus
from datetime import datetime

event = AuditEvent(
    event_id="evt-123",                    # Unique event ID
    event_type=AuditEventType.CREATE,      # Event type
    timestamp=datetime.now(),              # When it happened
    status=AuditStatus.SUCCESS,            # Operation status
    store_type="s3",                       # Store backend type
    store_id="my-bucket",                  # Store identifier
    item_id="run-456",                     # Item ID (optional)
    user_id="admin",                       # User ID (optional)
    session_id="sess-789",                 # Session ID (optional)
    duration_ms=45.2,                      # Duration in ms (optional)
    metadata={"region": "us-east-1"},      # Additional metadata
    error_message=None,                    # Error message if failed
    ip_address="192.168.1.1",              # Client IP (optional)
    user_agent="truthound/1.0",            # User agent (optional)
)

# Convert to dict
event_dict = event.to_dict()
```

### Data Redaction

실무 운영 가이드에서 Redact을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.audit import DataRedactor

redactor = DataRedactor(
    patterns=[
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    ],
    replacement="[REDACTED]",
    fields_to_redact=["email", "ssn", "password"],
)

# Redact event metadata
redacted_event = redactor.redact_event(event)

# Redact arbitrary data
redacted_data = redactor.redact({"email": "test@example.com"})
```

### 감사 Backends

#### InMemory감사Backend

```python
from truthound.stores.observability.audit import InMemoryAuditBackend

backend = InMemoryAuditBackend(max_events=10000)

# Log event
backend.log(event)

# Query events
events = backend.query(
    event_type=AuditEventType.CREATE,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    limit=100,
)

# Get all events
all_events = backend.get_all()

# Clear events
backend.clear()
```

#### File감사Backend

```python
from truthound.stores.observability.audit import FileAuditBackend

backend = FileAuditBackend(
    file_path=".truthound/audit.log",
    max_file_size_mb=100,
    max_files=10,  # Rotation
)

backend.log(event)
```

#### Json감사Backend

```python
from truthound.stores.observability.audit import JsonAuditBackend

backend = JsonAuditBackend(
    directory=".truthound/audit",
    file_prefix="audit",
    rotate_daily=True,
)

backend.log(event)

# Query from files
events = backend.query(
    event_type=AuditEventType.READ,
    start_time=datetime(2024, 1, 1),
)
```

#### Composite감사Backend

실무 운영 가이드에서 Log을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.audit import (
    CompositeAuditBackend,
    InMemoryAuditBackend,
    FileAuditBackend,
)

backend = CompositeAuditBackend(
    backends=[
        InMemoryAuditBackend(max_events=1000),
        FileAuditBackend(file_path=".truthound/audit.log"),
    ]
)

# Logs to all backends
backend.log(event)
```

#### Async감사Backend

Non-blocking 감사 logging:

```python
from truthound.stores.observability.audit import AsyncAuditBackend, InMemoryAuditBackend

backend = AsyncAuditBackend(
    backend=InMemoryAuditBackend(),
    queue_size=1000,
    flush_interval_seconds=5.0,
)

# Non-blocking log
backend.log(event)

# Flush pending events
backend.flush()

# Shutdown
backend.close()
```

### 감사Logger

High-level 감사 logging interface:

```python
from truthound.stores.observability.audit import AuditLogger, InMemoryAuditBackend

logger = AuditLogger(
    backend=InMemoryAuditBackend(),
    store_type="s3",
    store_id="my-bucket",
    redactor=DataRedactor(fields_to_redact=["password"]),
)

# Log operations
logger.log_create("run-123", user_id="admin")
logger.log_read("run-123", user_id="admin")
logger.log_update("run-123", user_id="admin")
logger.log_delete("run-123", user_id="admin")
logger.log_query({"status": "failure"}, user_id="admin")

# Log with status
logger.log_event(
    event_type=AuditEventType.REPLICATE,
    status=AuditStatus.SUCCESS,
    item_id="run-123",
    duration_ms=150.0,
    metadata={"target_region": "eu-west-1"},
)

# Log errors
logger.log_error(
    event_type=AuditEventType.CREATE,
    error_message="Connection timeout",
    item_id="run-123",
)
```

## 메트릭

### Metric Types

```python
from truthound.stores.observability.metrics import MetricType

MetricType.COUNTER    # Monotonically increasing counter
MetricType.GAUGE      # Value that can go up or down
MetricType.HISTOGRAM  # Distribution of values
MetricType.SUMMARY    # Statistical summary
```

### Metric Values

```python
from truthound.stores.observability.metrics import (
    MetricValue,
    HistogramValue,
    SummaryValue,
)

# Simple metric
metric = MetricValue(
    name="store_operations_total",
    value=42.0,
    labels={"store": "s3", "operation": "read"},
    timestamp=datetime.now(),
)

# Histogram
histogram = HistogramValue(
    name="store_operation_duration_seconds",
    count=100,
    sum=45.5,
    buckets={0.01: 10, 0.05: 50, 0.1: 80, 0.5: 95, 1.0: 100},
    labels={"store": "s3"},
)

# Summary
summary = SummaryValue(
    name="store_operation_latency",
    count=100,
    sum=45.5,
    quantiles={0.5: 0.4, 0.9: 0.8, 0.99: 1.2},
    labels={"store": "s3"},
)
```

### 메트릭 Backends

#### InMemory메트릭Backend

```python
from truthound.stores.observability.metrics import InMemoryMetricsBackend

backend = InMemoryMetricsBackend()

# Record metrics
backend.increment("operations_total", labels={"op": "read"})
backend.gauge("connections_active", 5, labels={"store": "s3"})
backend.histogram("latency_seconds", 0.05, labels={"op": "read"})
backend.summary("request_size_bytes", 1024, labels={"op": "write"})

# Get metrics
metrics = backend.get_metrics()

# Get specific metric
value = backend.get("operations_total", labels={"op": "read"})

# Reset
backend.reset()
```

#### Prometheus메트릭Backend

```python
from truthound.stores.observability.metrics import PrometheusMetricsBackend

backend = PrometheusMetricsBackend(
    prefix="truthound_store",
    default_labels={"service": "validation"},
)

# Record metrics (same interface)
backend.increment("operations_total", labels={"op": "read"})
backend.histogram("latency_seconds", 0.05, labels={"op": "read"})

# Start HTTP endpoint for scraping
backend.start_http_server(port=9090)

# Or push to gateway
backend.push_to_gateway(
    gateway_url="http://pushgateway:9091",
    job="truthound",
)

# Export in Prometheus format
prometheus_text = backend.export()
```

### 메트릭Registry

실무 운영 가이드에서 Singleton을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.metrics import MetricsRegistry

# Get singleton instance
registry = MetricsRegistry.get_instance()

# Register backend
registry.register_backend("prometheus", prometheus_backend)
registry.register_backend("memory", memory_backend)

# Record to all backends
registry.increment("operations_total", labels={"op": "read"})

# Get from primary backend
value = registry.get("operations_total")

# Export all
registry.export_all()
```

### Store메트릭

실무 운영 가이드에서 Helper을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.metrics import StoreMetrics

metrics = StoreMetrics(
    backend=prometheus_backend,
    store_type="s3",
    store_id="my-bucket",
)

# Track operations
metrics.record_operation("read", duration_ms=45.0, success=True)
metrics.record_operation("write", duration_ms=120.0, success=False)

# Track sizes
metrics.record_size(bytes_read=1024)
metrics.record_size(bytes_written=2048)

# Track connections
metrics.record_connection_opened()
metrics.record_connection_closed()

# Track cache
metrics.record_cache_hit()
metrics.record_cache_miss()

# Track errors
metrics.record_error("ConnectionTimeout")
```

### Standard 메트릭

Store메트릭 records these metrics:

| 실무 운영 가이드에서 Metric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Labels을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|--------|-------------|
| 실무 운영 가이드에서 `store_operations_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `store_operation_duration_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Histogram을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Operation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `store_bytes_read_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `store_bytes_written_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Total을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `store_connections_active`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Gauge을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Active을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `store_cache_hits_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 hits |
| 실무 운영 가이드에서 `store_cache_misses_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 misses |
| 실무 운영 가이드에서 `store_errors_total`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Tracing

### Span Kinds

```python
from truthound.stores.observability.tracing import SpanKind

SpanKind.INTERNAL   # Internal operation
SpanKind.SERVER     # Server-side operation
SpanKind.CLIENT     # Client-side operation (e.g., S3 call)
SpanKind.PRODUCER   # Message producer
SpanKind.CONSUMER   # Message consumer
```

### Span Status

```python
from truthound.stores.observability.tracing import SpanStatus

SpanStatus.UNSET    # Status not set
SpanStatus.OK       # Operation succeeded
SpanStatus.ERROR    # Operation failed
```

### SpanContext

```python
from truthound.stores.observability.tracing import SpanContext

# Create context
context = SpanContext(
    trace_id="abc123def456",
    span_id="span789",
    parent_span_id="parent123",
    trace_flags=1,  # Sampled
    trace_state={"vendor": "value"},
)

# Parse W3C traceparent
context = SpanContext.from_traceparent(
    "00-abc123def456-span789-01"
)

# Export as traceparent
traceparent = context.to_traceparent()
# "00-abc123def456-span789-01"
```

### Span

```python
from truthound.stores.observability.tracing import Span, SpanKind, SpanStatus

# Create span
span = Span(
    name="store.read",
    kind=SpanKind.CLIENT,
    context=context,
)

# Add attributes
span.set_attribute("store.type", "s3")
span.set_attribute("item.id", "run-123")

# Add events
span.add_event("cache.miss", {"key": "run-123"})
span.add_event("retry.attempt", {"attempt": 1})

# Set status
span.set_status(SpanStatus.OK)

# End span
span.end()

# Get duration
print(f"Duration: {span.duration_ms}ms")
```

### Tracers

#### NoopTracer

실무 운영 가이드에서 No-op을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.tracing import NoopTracer

tracer = NoopTracer()

# All operations are no-ops
with tracer.start_span("operation") as span:
    span.set_attribute("key", "value")  # Does nothing
```

#### InMemoryTracer

실무 운영 가이드에서 In-memory을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.tracing import InMemoryTracer

tracer = InMemoryTracer(max_spans=1000)

# Create spans
with tracer.start_span("parent_operation") as parent:
    parent.set_attribute("key", "value")

    with tracer.start_span("child_operation") as child:
        child.set_attribute("nested", True)

# Get recorded spans
spans = tracer.get_spans()
print(f"Recorded {len(spans)} spans")

# Get spans by name
read_spans = tracer.get_spans_by_name("store.read")

# Clear spans
tracer.clear()
```

#### OpenTelemetryTracer

실무 운영 가이드에서 Production, OpenTelemetry을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.tracing import OpenTelemetryTracer

# OTLP exporter (default)
tracer = OpenTelemetryTracer(
    service_name="truthound",
    endpoint="http://localhost:4317",
    exporter_type="otlp",
)

# Jaeger exporter
tracer = OpenTelemetryTracer(
    service_name="truthound",
    endpoint="http://localhost:14268/api/traces",
    exporter_type="jaeger",
)

# Zipkin exporter
tracer = OpenTelemetryTracer(
    service_name="truthound",
    endpoint="http://localhost:9411/api/v2/spans",
    exporter_type="zipkin",
)

# Use tracer
with tracer.start_span("store.read", kind=SpanKind.CLIENT) as span:
    span.set_attribute("store.type", "s3")
    span.set_attribute("item.id", "run-123")
    # ... perform operation
    span.set_status(SpanStatus.OK)
```

### Tracer Factory

```python
from truthound.stores.observability.tracing import Tracer

# Get tracer (auto-selects based on config)
tracer = Tracer.create(
    service_name="truthound",
    enabled=True,
    exporter_type="otlp",
    endpoint="http://localhost:4317",
)

# Or create disabled tracer
tracer = Tracer.create(enabled=False)  # Returns NoopTracer
```

### Context Propagation

```python
# Extract context from headers
incoming_context = SpanContext.from_traceparent(
    headers.get("traceparent")
)

# Create span with parent context
with tracer.start_span("operation", parent=incoming_context) as span:
    # ... perform operation

    # Propagate context to downstream
    outgoing_headers = {
        "traceparent": span.context.to_traceparent()
    }
```

## Combined 관측성

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound.stores.observability.audit import AuditLogger, InMemoryAuditBackend
from truthound.stores.observability.metrics import StoreMetrics, PrometheusMetricsBackend
from truthound.stores.observability.tracing import OpenTelemetryTracer, SpanKind

# Setup
audit = AuditLogger(
    backend=InMemoryAuditBackend(),
    store_type="s3",
    store_id="my-bucket",
)
metrics = StoreMetrics(
    backend=PrometheusMetricsBackend(),
    store_type="s3",
    store_id="my-bucket",
)
tracer = OpenTelemetryTracer(
    service_name="truthound",
    endpoint="http://localhost:4317",
)

# Instrumented operation
def get_result(item_id: str) -> dict:
    with tracer.start_span("store.read", kind=SpanKind.CLIENT) as span:
        span.set_attribute("item.id", item_id)

        start = time.time()
        try:
            result = store.get(item_id)
            duration_ms = (time.time() - start) * 1000

            # Record success
            audit.log_read(item_id, duration_ms=duration_ms)
            metrics.record_operation("read", duration_ms=duration_ms, success=True)
            span.set_status(SpanStatus.OK)

            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000

            # Record failure
            audit.log_error(AuditEventType.READ, str(e), item_id=item_id)
            metrics.record_operation("read", duration_ms=duration_ms, success=False)
            metrics.record_error(type(e).__name__)
            span.set_status(SpanStatus.ERROR, str(e))

            raise
```

## 권장 방식

### 감사 로깅

- 실무 운영 가이드에서 `DataRedactor`, DataRedactor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Rotate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Query을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 메트릭

- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Keep, IDs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Export, Prometheus을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Tracing

- 실무 운영 가이드에서 Propagate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Add을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 실무 운영 가이드에서 FileSystem, Store, Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Cloud, Storage, GCS, Azure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Replication, Cross-region을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
