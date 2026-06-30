# 메트릭 설정

실무 운영 가이드에서 Truthound, Prometheus-compatible을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

## 메트릭Config

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

## 환경 변수

| 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설정 Field | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|--------------|---------|
| 실무 운영 가이드에서 `METRICS_ENABLED`, METRICS_ENABLED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `enabled`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `true`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SERVICE_NAME`, SERVICE_NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `service`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ENVIRONMENT`, ENVIRONMENT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `environment`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `METRICS_HTTP_ENABLED`, METRICS_HTTP_ENABLED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `enable_http`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `METRICS_PORT`, METRICS_PORT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `port`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `9090`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `METRICS_PUSH_GATEWAY_URL`, METRICS_PUSH_GATEWAY_URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `push_gateway_url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `""`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 검증기 메트릭

메트릭 for 검증 operations:

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

## 체크포인트 메트릭

메트릭 for 체크포인트 operations:

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

## DataSource 메트릭

메트릭 for data 소스 operations:

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

## HTTP 메트릭 Server

실무 운영 가이드에서 HTTP을(를) 다루는 항목입니다:

```python
from truthound.infrastructure.metrics import MetricsServer

server = MetricsServer(port=9090, path="/metrics")
server.start()

# Access metrics at http://localhost:9090/metrics
# Server runs in background thread
```

## Push Gateway

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
configure_metrics(
    enabled=True,
    push_gateway_url="http://pushgateway:9091",
    push_interval=15.0,
    push_job="truthound",
)
```

## Store 관측성 메트릭

실무 운영 가이드에서 Additional을(를) 다루는 항목입니다:

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

## Prometheus Scrape 설정

Example Prometheus 설정:

```yaml
scrape_configs:
  - job_name: 'truthound'
    static_configs:
      - targets: ['truthound:9090']
    scrape_interval: 15s
    metrics_path: /metrics
```

## Grafana Dashboard Example

실무 운영 가이드에서 Key을(를) 다루는 항목입니다:

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
