---
title: Metrics & Tracing
---

# Metrics & Tracing

Provides a metrics system for application performance measurement and distributed tracing.

## Metrics

### Basic Usage

```python
from common.metrics import Counter, Gauge, Histogram, Summary

# Counter: monotonically increasing value
requests = Counter("requests_total", "Total request count")
requests.inc()
requests.inc(5, labels={"method": "POST", "endpoint": "/api/check"})

# Gauge: value that can increase or decrease
active_connections = Gauge("active_connections", "Active connections")
active_connections.set(10)
active_connections.inc()
active_connections.dec()

# Histogram: distribution measurement
latency = Histogram(
    "request_duration_seconds",
    "Request duration",
    buckets=(0.1, 0.5, 1.0, 5.0),
)
latency.observe(0.42)

# Time measurement context
with latency.time():
    process_request()

# Summary: quantile calculation
response_size = Summary(
    "response_size_bytes",
    "Response size",
    quantiles=(0.5, 0.9, 0.99),
)
response_size.observe(1024)
```

### Preset Configurations

```python
from common.metrics import (
    DEFAULT_METRICS_CONFIG,           # Default configuration
    DISABLED_METRICS_CONFIG,          # Disabled
    HIGH_CARDINALITY_METRICS_CONFIG,  # Fine-grained buckets
)
```

### Decorators

```python
from common.metrics import timed, counted, instrumented

# Automatic function execution time measurement
@timed("process_duration_seconds")
def process_data(data):
    return validate(data)

# Automatic function call counting
@counted("api_calls_total", labels={"endpoint": "/check"})
def api_call():
    return requests.get("/data")

# Combined metrics + tracing
@instrumented(labels={"component": "validator"})
def validate(data):
    return truthound.check(data)
```

### Registry Usage

```python
from common.metrics import (
    MetricsRegistry,
    configure_metrics,
    counter,
    gauge,
    histogram,
)

# Configure global registry
configure_metrics(
    config=MetricsConfig(prefix="myapp"),
    exporters=[ConsoleMetricExporter()],
)

# Create metrics with convenience functions
c = counter("requests", "Request count")
g = gauge("memory", "Memory usage")
h = histogram("latency", "Latency")

# Collect and export metrics
registry = MetricsRegistry()
all_metrics = registry.collect()
registry.export()
```

### Exporters

```python
from common.metrics import (
    ConsoleMetricExporter,     # Console output
    InMemoryMetricExporter,    # Memory storage (for testing)
    CompositeMetricExporter,   # Multiple exporter combination
)

# Verify metrics in tests
exporter = InMemoryMetricExporter()
configure_metrics(exporters=[exporter])

process_data()
registry.export()
assert len(exporter.metrics) > 0
```

## Tracing

### Basic Usage

```python
from common.metrics import Span, SpanKind

# Basic span creation
with Span("process_data") as span:
    span.set_attribute("rows", 1000)
    span.set_attribute("source", "s3")
    result = process(data)
    span.add_event("processing_complete")

# Nested spans
with Span("parent_operation") as parent:
    with Span("child_operation", parent=parent) as child:
        child.set_attribute("step", 1)
        do_work()
```

### Decorators

```python
from common.metrics import trace

# Automatic function execution tracing
@trace(name="validate_data")
def validate(data):
    return truthound.check(data)

# Specify SpanKind
@trace(name="fetch_external", kind=SpanKind.CLIENT)
def fetch_data():
    return api.get("/data")

# Add attributes
@trace(name="process", attributes={"component": "validator"})
def process(data):
    return transform(data)
```

### Context Propagation (W3C Trace Context)

```python
from common.metrics import inject_context, extract_context, TraceContext

# Inject context into HTTP request
headers = {}
inject_context(headers)
# headers = {"traceparent": "00-abc123...-def456...-01"}

# Extract context from response
ctx = extract_context(incoming_headers)
if ctx:
    print(f"Trace ID: {ctx.trace_id}")
    print(f"Span ID: {ctx.span_id}")

# Manual context creation
ctx = TraceContext(
    trace_id="0af7651916cd43dd8448eb211c80319c",
    span_id="b7ad6b7169203331",
    sampled=True,
)
headers = ctx.to_headers()
```

### Span Exporters

```python
from common.metrics import (
    ConsoleSpanExporter,      # Console output
    InMemorySpanExporter,     # Memory storage (for testing)
    CompositeSpanExporter,    # Multiple exporter combination
)
```

### Exception Recording

```python
from common.metrics import Span, SpanStatus

with Span("risky_operation") as span:
    try:
        risky_function()
    except Exception as e:
        span.record_exception(e)
        span.set_status(SpanStatus.ERROR, str(e))
        raise
```

## Prometheus Export

Metrics export for Prometheus integration:

```python
from common.exporters.prometheus import (
    PrometheusExporter,
    create_prometheus_exporter,
    create_pushgateway_exporter,
    create_prometheus_http_server,
)

# Create simple exporter
exporter = create_prometheus_exporter(
    namespace="myapp",
    job_name="data_quality",
    const_labels={"env": "production"},
)

# Push Gateway usage
exporter = create_pushgateway_exporter(
    gateway_url="http://pushgateway:9091",
    job_name="batch_job",
)

# HTTP server (for scraping)
server = create_prometheus_http_server(
    host="0.0.0.0",
    port=9090,
    path="/metrics",
)
server.start()
```
