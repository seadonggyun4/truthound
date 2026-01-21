# Metrics Configuration

Truthound provides Prometheus-compatible metrics for monitoring validators, checkpoints, and data sources.

## Quick Start

```python
from truthound.infrastructure.metrics import configure_metrics, get_metrics

# Configure metrics
configure_metrics(
    enabled=True,
    service="truthound",
    environment="production",
    enable_http=True,
    port=9090,
)

# Access metrics at http://localhost:9090/metrics
```

## MetricsConfig

```python
from truthound.infrastructure.metrics import MetricsConfig

config = MetricsConfig(
    enabled=True,                    # Enable metrics collection
    service="truthound",             # Service name for labels
    environment="production",        # Environment name for labels
    namespace="truthound",           # Metric name prefix

    # HTTP server
    enable_http=False,               # Enable HTTP endpoint
    host="0.0.0.0",                  # HTTP server host
    port=9090,                       # HTTP server port
    path="/metrics",                 # Metrics endpoint path

    # Push gateway
    push_gateway_url="",             # Prometheus push gateway URL
    push_interval=15.0,              # Push interval in seconds
    push_job="truthound",            # Job name for push gateway

    # Default labels
    default_labels={},               # Labels added to all metrics

    # Histogram buckets
    latency_buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ),
    size_buckets=(
        100, 1000, 10000, 100000, 1000000,
        10000000, 100000000, 1000000000,
    ),
)
```

## Environment Variables

| Variable | Config Field | Default |
|----------|--------------|---------|
| `METRICS_ENABLED` | `enabled` | `true` |
| `SERVICE_NAME` | `service` | `""` |
| `ENVIRONMENT` | `environment` | `""` |
| `METRICS_HTTP_ENABLED` | `enable_http` | `false` |
| `METRICS_PORT` | `port` | `9090` |
| `METRICS_PUSH_GATEWAY_URL` | `push_gateway_url` | `""` |

## Validator Metrics

Metrics for validation operations:

```
truthound_validator_executions_total
  {validator, dataset, column, status}

truthound_validator_duration_seconds
  {validator, dataset}

truthound_validator_issues_total
  {validator, dataset, column, severity}

truthound_validator_rows_processed_total
  {validator, dataset}

truthound_validators_active
  {dataset}
```

### Usage

```python
from truthound.infrastructure.metrics import ValidatorMetrics

metrics = ValidatorMetrics()

# Record execution
metrics.record_execution(
    validator="not_null",
    dataset="users",
    column="email",
    success=True,
    duration_seconds=0.5,
    issues_found=0,
    rows_processed=10000,
)

# Time context manager
with metrics.time("not_null", "users", "email"):
    run_validation()

# Manual metric updates
metrics.validations_total.inc(validator="null", status="success")
metrics.issues_found.inc(severity="high", count=15)
```

## Checkpoint Metrics

Metrics for checkpoint operations:

```
truthound_checkpoint_executions_total
  {checkpoint, status}

truthound_checkpoint_duration_seconds
  {checkpoint}

truthound_checkpoint_issues
  {checkpoint, severity}

truthound_checkpoint_validators_run_total
  {checkpoint}

truthound_checkpoint_last_execution_timestamp
  {checkpoint}

truthound_checkpoints_running
```

### Usage

```python
from truthound.infrastructure.metrics import CheckpointMetrics

metrics = CheckpointMetrics()

# Record checkpoint execution
metrics.checkpoints_total.inc(name="daily_check", status="success")
metrics.checkpoint_duration.observe(duration_seconds)
metrics.actions_total.inc(action_type="slack", status="success")
```

## DataSource Metrics

Metrics for data source operations:

```
truthound_datasource_connections_total
  {backend, status}

truthound_datasource_query_duration_seconds
  {backend}

truthound_datasource_rows_processed_total
  {backend}
```

### Usage

```python
from truthound.infrastructure.metrics import DataSourceMetrics

metrics = DataSourceMetrics()

metrics.connections_total.inc(backend="postgresql", status="success")
metrics.query_duration.observe(duration_seconds, backend="postgresql")
metrics.rows_processed.inc(count=100000)
```

## HTTP Metrics Server

Start an HTTP server to expose metrics:

```python
from truthound.infrastructure.metrics import MetricsServer

server = MetricsServer(port=9090, path="/metrics")
server.start()

# Access metrics at http://localhost:9090/metrics
# Server runs in background thread
```

## Push Gateway

For environments where pull-based scraping isn't possible:

```python
configure_metrics(
    enabled=True,
    push_gateway_url="http://pushgateway:9091",
    push_interval=15.0,
    push_job="truthound",
)
```

## Store Observability Metrics

Additional metrics for storage operations:

```python
from truthound.stores.observability import MetricsConfig, MetricsExportFormat

config = MetricsConfig(
    enabled=True,
    export_format=MetricsExportFormat.PROMETHEUS,
    prefix="truthound_store",
    labels={"service": "data-quality"},
    enable_http_server=True,
    http_port=9090,
    http_path="/metrics",
    push_gateway_url=None,
    push_interval_seconds=10.0,
    include_timestamps=True,
)
```

## Prometheus Scrape Configuration

Example Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'truthound'
    static_configs:
      - targets: ['truthound:9090']
    scrape_interval: 15s
    metrics_path: /metrics
```

## Grafana Dashboard Example

Key metrics to display:

```
# Validation success rate
sum(rate(truthound_validator_executions_total{status="success"}[5m])) /
sum(rate(truthound_validator_executions_total[5m]))

# Issues by severity
sum by (severity) (truthound_validator_issues_total)

# Checkpoint duration p95
histogram_quantile(0.95, rate(truthound_checkpoint_duration_seconds_bucket[5m]))

# Active validators
truthound_validators_active

# Rows processed per second
rate(truthound_validator_rows_processed_total[1m])
```
