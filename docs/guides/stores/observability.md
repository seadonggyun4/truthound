# Store Observability

Comprehensive observability for storage operations: audit logging, Prometheus metrics, and OpenTelemetry tracing.

## Overview

The observability module provides three pillars of observability:

- **Audit Logging** - Track all storage operations with detailed event logging
- **Metrics** - Prometheus-compatible metrics for monitoring and alerting
- **Tracing** - Distributed tracing with OpenTelemetry support

## Audit Logging

### Audit Event Types

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

### Audit Status

```python
from truthound.stores.observability.audit import AuditStatus

AuditStatus.SUCCESS    # Operation succeeded
AuditStatus.FAILURE    # Operation failed
AuditStatus.PARTIAL    # Partial success
AuditStatus.DENIED     # Access denied
```

### AuditEvent

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

Redact sensitive data before logging:

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

### Audit Backends

#### InMemoryAuditBackend

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

#### FileAuditBackend

```python
from truthound.stores.observability.audit import FileAuditBackend

backend = FileAuditBackend(
    file_path=".truthound/audit.log",
    max_file_size_mb=100,
    max_files=10,  # Rotation
)

backend.log(event)
```

#### JsonAuditBackend

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

#### CompositeAuditBackend

Log to multiple backends:

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

#### AsyncAuditBackend

Non-blocking audit logging:

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

### AuditLogger

High-level audit logging interface:

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

## Metrics

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

### Metrics Backends

#### InMemoryMetricsBackend

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

#### PrometheusMetricsBackend

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

### MetricsRegistry

Singleton registry for global metric access:

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

### StoreMetrics

Helper class for common store metrics:

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

### Standard Metrics

StoreMetrics records these metrics:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `store_operations_total` | Counter | store, operation, status | Total operations |
| `store_operation_duration_seconds` | Histogram | store, operation | Operation latency |
| `store_bytes_read_total` | Counter | store | Total bytes read |
| `store_bytes_written_total` | Counter | store | Total bytes written |
| `store_connections_active` | Gauge | store | Active connections |
| `store_cache_hits_total` | Counter | store | Cache hits |
| `store_cache_misses_total` | Counter | store | Cache misses |
| `store_errors_total` | Counter | store, error_type | Error count |

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

No-op tracer for disabled tracing:

```python
from truthound.stores.observability.tracing import NoopTracer

tracer = NoopTracer()

# All operations are no-ops
with tracer.start_span("operation") as span:
    span.set_attribute("key", "value")  # Does nothing
```

#### InMemoryTracer

In-memory tracer for testing:

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

Production tracer with OpenTelemetry:

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

## Combined Observability

Use all three pillars together:

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

## Best Practices

### Audit Logging

- Use `DataRedactor` to remove sensitive data
- Enable async logging for high-throughput scenarios
- Rotate log files to manage storage
- Query audit logs for compliance reporting

### Metrics

- Use consistent label values
- Keep cardinality low (avoid unique IDs as labels)
- Set up alerts on error rates and latency
- Export to Prometheus for visualization

### Tracing

- Propagate context across service boundaries
- Add meaningful attributes to spans
- Use appropriate span kinds
- Sample in production to reduce overhead

## Next Steps

- [FileSystem Store](filesystem.md) - Local storage
- [Cloud Storage](cloud-storage.md) - S3, GCS, Azure
- [Replication](replication.md) - Cross-region replication
