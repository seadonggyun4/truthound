# Distributed Processing

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `src/truthound/profiler/distributed.py`, Spark, Dask, Ray, Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## DistributedBackend

```python
class DistributedBackend(str, Enum):
    """Distributed processing backends"""

    LOCAL = "local"    # Local multithread/multiprocess
    SPARK = "spark"    # Apache Spark
    DASK = "dask"      # Dask Distributed
    RAY = "ray"        # Ray
    AUTO = "auto"      # Automatic detection
```

## PartitionStrategy

```python
class PartitionStrategy(str, Enum):
    """Partition strategies"""

    ROW_BASED = "row_based"       # Row-based partitioning
    COLUMN_BASED = "column_based" # Column-based partitioning
    HYBRID = "hybrid"             # Hybrid (row + column)
    HASH = "hash"                 # Hash-based partitioning
```

## Backend-Specific 설정

### SparkConfig

```python
@dataclass
class SparkConfig:
    """Spark configuration"""

    app_name: str = "truthound-profiler"
    master: str = "local[*]"
    executor_memory: str = "4g"
    executor_cores: int = 2
    num_executors: int = 4
    spark_config: dict[str, str] = field(default_factory=dict)
```

### DaskConfig

```python
@dataclass
class DaskConfig:
    """Dask configuration"""

    scheduler: str = "threads"  # threads, processes, synchronous
    num_workers: int | None = None  # None = number of CPU cores
    memory_limit: str = "4GB"
    dashboard_address: str = ":8787"
```

### RayConfig

```python
@dataclass
class RayConfig:
    """Ray configuration"""

    address: str | None = None  # None = local cluster
    num_cpus: int | None = None
    num_gpus: int | None = None
    object_store_memory: int | None = None
```

## Distributed프로파일러

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import DistributedProfiler, DistributedBackend

# Automatic backend selection
profiler = DistributedProfiler.create(backend=DistributedBackend.AUTO)

# Spark backend
profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=SparkConfig(master="spark://master:7077"),
)

# Dask backend
profiler = DistributedProfiler.create(
    backend=DistributedBackend.DASK,
    config=DaskConfig(scheduler="distributed"),
)

# Execute profiling
profile = profiler.profile("hdfs://data/large.parquet")
```

## LocalBackend

실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import LocalBackend

backend = LocalBackend(
    num_workers=4,
    use_processes=True,  # True = processes, False = threads
)

# Parallel column profiling
results = backend.profile_columns_parallel(lf, columns)
```

## SparkBackend

실무 운영 가이드에서 Apache, Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import SparkBackend, SparkConfig

config = SparkConfig(
    master="spark://master:7077",
    executor_memory="8g",
    num_executors=10,
)

backend = SparkBackend(config)

# Profile Spark DataFrame
profile = backend.profile(spark_df)

# Profile HDFS file
profile = backend.profile_file("hdfs://data/large.parquet")
```

## DaskBackend

실무 운영 가이드에서 Dask, Distributed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import DaskBackend, DaskConfig

config = DaskConfig(
    scheduler="distributed",
    num_workers=8,
    memory_limit="8GB",
)

backend = DaskBackend(config)

# Profile Dask DataFrame
profile = backend.profile(dask_df)

# Profile partitioned files
profile = backend.profile_file("s3://bucket/data/*.parquet")
```

## RayBackend

실무 운영 가이드에서 Ray을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import RayBackend, RayConfig

config = RayConfig(
    address="ray://cluster:10001",
    num_cpus=16,
)

backend = RayBackend(config)

# Profile Ray Dataset
profile = backend.profile(ray_dataset)
```

## BackendRegistry

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.distributed import BackendRegistry

# List registered backends
backends = BackendRegistry.list()  # ["local", "spark", "dask", "ray"]

# Create backend instance
backend = BackendRegistry.create("spark", config=spark_config)

# Register custom backend
@BackendRegistry.register("my_backend")
class MyCustomBackend:
    def profile(self, data):
        # Custom profiling logic
        pass
```

## Partition Strategies

### ROW_BASED - Row-Based

```python
from truthound.profiler.distributed import RowBasedPartitioner

partitioner = RowBasedPartitioner(num_partitions=10)
partitions = partitioner.partition(lf)

# Each partition contains 1/10 of total rows
```

### COLUMN_BASED - 컬럼-Based

```python
from truthound.profiler.distributed import ColumnBasedPartitioner

partitioner = ColumnBasedPartitioner()
column_groups = partitioner.partition(lf, num_workers=4)

# Each worker processes 1/4 of columns
```

### HYBRID - Hybrid

```python
from truthound.profiler.distributed import HybridPartitioner

partitioner = HybridPartitioner(
    row_partitions=4,
    column_partitions=2,
)
# 4 x 2 = 8 partitions
```

## Automatic Backend Selection

```python
from truthound.profiler.distributed import auto_select_backend

# Automatic selection based on environment
backend = auto_select_backend()

# Selection logic:
# 1. Spark session exists → Spark
# 2. Dask client exists → Dask
# 3. Ray initialized → Ray
# 4. Default → Local
```

## 결과 Merging

Merge distributed processing 결과.

```python
from truthound.profiler.distributed import ProfileMerger

merger = ProfileMerger()

# Merge partition profiles
partial_profiles = [profile1, profile2, profile3, profile4]
merged = merger.merge(partial_profiles)

# Statistics merge strategy:
# - count: sum
# - mean: weighted average
# - min/max: global min/max
# - quantiles: approximate merge
```

## 모니터링

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 개요

```python
from truthound.profiler.distributed.monitoring import (
    DistributedMonitor,
    MonitorConfig,
    MonitorFactory,
)

# Quick start with factory presets
monitor = MonitorFactory.with_console()

# Or create with full configuration
config = MonitorConfig.production()
monitor = DistributedMonitor(config)

# Use with profiler
profiler = DistributedProfiler.create(
    backend="spark",
    monitor=monitor,
)

profile = profiler.profile(data)
```

### MonitorConfig

실무 운영 가이드에서 Configuration을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import MonitorConfig

# Minimal overhead - basic tracking only
config = MonitorConfig.minimal()

# Standard features (default)
config = MonitorConfig.standard()

# All features enabled
config = MonitorConfig.full()

# Production-optimized
config = MonitorConfig.production()
```

#### 설정 Options

```python
@dataclass
class MonitorConfig:
    enable_monitoring: bool = True
    enable_task_tracking: bool = True
    enable_progress_aggregation: bool = True
    enable_health_monitoring: bool = True
    enable_metrics_collection: bool = True
    task_tracker: TaskTrackerConfig = ...
    health_check: HealthCheckConfig = ...
    metrics: MetricsConfig = ...
    callbacks: CallbackConfig = ...
    log_level: str = "INFO"
```

### 태스크 Tracking

실무 운영 가이드에서 Track을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import TaskTracker, TaskTrackerConfig

config = TaskTrackerConfig(
    max_history_size=1000,
    timeout_seconds=3600,
    max_retries=3,
    enable_progress_tracking=True,
)

tracker = TaskTracker(config=config)

# Submit and track tasks
task_id = tracker.submit_task(
    task_type="profile_partition",
    partition_id=0,
    metadata={"rows": 1000000},
)

tracker.start_task(task_id, worker_id="worker-1")
tracker.update_progress(task_id, progress=0.5, rows_processed=500000)
tracker.complete_task(task_id, result={"stats": {...}})

# Query task state
task = tracker.get_task(task_id)
print(f"Status: {task.state}, Duration: {task.duration_seconds}s")
```

### Progress Aggregation

실무 운영 가이드에서 Aggregate을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import (
    DistributedProgressAggregator,
    StreamingProgressAggregator,
)

# Basic aggregator
aggregator = DistributedProgressAggregator(
    on_progress=lambda p: print(f"Overall: {p.percent:.1f}%"),
    milestone_interval=10,  # Emit event every 10%
)

aggregator.set_total_partitions(10)
aggregator.start_partition(0, total_rows=1000000)
aggregator.update_partition(0, progress=0.5, rows_processed=500000)
aggregator.complete_partition(0, rows_processed=1000000)

# Get aggregated progress
progress = aggregator.get_progress()
print(f"Completed: {progress.completed_partitions}/{progress.total_partitions}")
print(f"ETA: {progress.estimated_remaining_seconds:.0f}s")
print(f"Throughput: {progress.rows_per_second:.0f} rows/s")

# Streaming aggregator for real-time UI
streaming = StreamingProgressAggregator(
    on_progress=update_ui,
    interpolation_interval_ms=100,
)
```

### Health 모니터링

실무 운영 가이드에서 Monitor을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import (
    WorkerHealthMonitor,
    HealthCheckConfig,
    HealthStatus,
)

config = HealthCheckConfig(
    heartbeat_timeout_seconds=30,
    stall_threshold_seconds=60,
    memory_warning_percent=80,
    memory_critical_percent=95,
    cpu_warning_percent=90,
    error_rate_warning=0.1,
    error_rate_critical=0.25,
)

monitor = WorkerHealthMonitor(
    config=config,
    on_event=lambda e: print(f"Health: {e.message}"),
)

# Register and track workers
monitor.register_worker("worker-1")
monitor.record_heartbeat(
    "worker-1",
    cpu_percent=65,
    memory_percent=70,
    active_tasks=3,
)
monitor.record_task_complete("worker-1", duration_seconds=2.5, success=True)

# Check health
health = monitor.get_worker_health("worker-1")
print(f"Status: {health.status}")  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL

# Get overall system health
overall = monitor.get_overall_health()
unhealthy = monitor.get_unhealthy_workers()
stalled = monitor.check_stalled_workers()
```

### 메트릭 Collection

실무 운영 가이드에서 Collect을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import (
    DistributedMetricsCollector,
    MetricsConfig,
)

config = MetricsConfig(
    enable_metrics=True,
    collection_interval_seconds=5.0,
    enable_percentiles=True,
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    enable_prometheus=True,
    prometheus_port=9090,
)

collector = DistributedMetricsCollector(config=config)

# Record metrics
collector.record_task_duration("profile_partition", 2.5)
collector.record_partition_progress(0, 0.5)
collector.record_worker_metric("worker-1", "cpu_percent", 65)
collector.increment_counter("tasks_completed")

# Get metrics
metrics = collector.get_metrics()
print(f"Tasks: {metrics.tasks_completed}")
print(f"Avg duration: {metrics.avg_task_duration_seconds:.2f}s")
print(f"P99 duration: {metrics.task_duration_p99:.2f}s")

# Export to Prometheus format
prometheus_output = collector.export_prometheus()
```

### Callback System

실무 운영 가이드에서 Flexible을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import (
    ConsoleMonitorCallback,
    ProgressBarCallback,
    LoggingMonitorCallback,
    FileMonitorCallback,
    WebhookMonitorCallback,
    CallbackChain,
    AsyncMonitorCallback,
)

# Console output with colors
console = ConsoleMonitorCallback(
    show_progress=True,
    show_events=True,
    use_colors=True,
)

# Progress bar (tqdm)
progress_bar = ProgressBarCallback(
    desc="Profiling",
    unit="partitions",
)

# Structured logging
logging_cb = LoggingMonitorCallback(
    logger_name="truthound.monitor",
    log_level=logging.INFO,
)

# File output (JSONL)
file_cb = FileMonitorCallback(
    output_path="monitoring.jsonl",
    format="jsonl",  # or "json"
)

# Webhook notifications
webhook = WebhookMonitorCallback(
    url="https://hooks.slack.com/...",
    headers={"Authorization": "Bearer token"},
    event_filter=lambda e: e.severity >= EventSeverity.WARNING,
)

# Combine callbacks
chain = CallbackChain([console, logging_cb, file_cb])

# Async callback for non-blocking operations
async_cb = AsyncMonitorCallback(
    callback=webhook,
    buffer_size=100,
)
```

### Monitor Factory

실무 운영 가이드에서 Create을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import MonitorFactory

# Minimal - basic tracking only
monitor = MonitorFactory.minimal()

# Console output
monitor = MonitorFactory.with_console(
    show_progress=True,
    use_colors=True,
)

# Logging output
monitor = MonitorFactory.with_logging(
    logger_name="my_app.profiler",
    log_level=logging.INFO,
)

# Production setup
monitor = MonitorFactory.production(
    log_path="/var/log/truthound/monitor.jsonl",
    webhook_url="https://alerts.example.com/webhook",
)
```

### Monitor Registry

실무 운영 가이드에서 Register을(를) 다루는 항목입니다:

```python
from truthound.profiler.distributed.monitoring import MonitorRegistry

# Register custom callback
@MonitorRegistry.register_callback("my_callback")
class MyCallback(MonitorCallbackAdapter):
    def on_event(self, event: MonitorEvent) -> None:
        # Custom handling
        pass

# Register custom preset
@MonitorRegistry.register_preset("my_preset")
def my_preset(**kwargs) -> DistributedMonitor:
    config = MonitorConfig(...)
    return DistributedMonitor(config)

# Use registered components
monitor = MonitorRegistry.create_monitor("my_preset")
callback = MonitorRegistry.create_callback("my_callback")
```

### Backend Adapters

Backend-specific 모니터링 adapters:

```python
from truthound.profiler.distributed.monitoring.adapters import (
    LocalMonitorAdapter,
    DaskMonitorAdapter,
    SparkMonitorAdapter,
    RayMonitorAdapter,
)

# Local backend adapter
local_adapter = LocalMonitorAdapter(
    executor=thread_pool_executor,
    poll_interval_seconds=1.0,
)

# Dask adapter
dask_adapter = DaskMonitorAdapter(
    client=dask_client,
    poll_interval_seconds=2.0,
)
dask_adapter.connect()
workers = dask_adapter.get_workers()
metrics = dask_adapter.get_metrics()

# Spark adapter
spark_adapter = SparkMonitorAdapter(
    spark_session=spark,
    poll_interval_seconds=5.0,
)

# Ray adapter
ray_adapter = RayMonitorAdapter(
    address="ray://cluster:10001",
    poll_interval_seconds=5.0,
)
```

### Event Types

Available 모니터링 event types:

```python
from truthound.profiler.distributed.monitoring import MonitorEventType

# Task lifecycle
MonitorEventType.TASK_SUBMITTED
MonitorEventType.TASK_STARTED
MonitorEventType.TASK_PROGRESS
MonitorEventType.TASK_COMPLETED
MonitorEventType.TASK_FAILED
MonitorEventType.TASK_RETRYING

# Partition events
MonitorEventType.PARTITION_START
MonitorEventType.PARTITION_COMPLETE
MonitorEventType.PARTITION_ERROR

# Progress events
MonitorEventType.PROGRESS_UPDATE
MonitorEventType.PROGRESS_MILESTONE
MonitorEventType.AGGREGATION_COMPLETE

# Worker events
MonitorEventType.WORKER_REGISTERED
MonitorEventType.WORKER_UNREGISTERED
MonitorEventType.WORKER_HEALTHY
MonitorEventType.WORKER_UNHEALTHY
MonitorEventType.WORKER_STALLED
MonitorEventType.WORKER_RECOVERED

# Monitor lifecycle
MonitorEventType.MONITOR_START
MonitorEventType.MONITOR_STOP
```

### Complete Example

```python
from truthound.profiler.distributed import (
    DistributedProfiler,
    DistributedBackend,
    SparkConfig,
)
from truthound.profiler.distributed.monitoring import (
    DistributedMonitor,
    MonitorConfig,
    ConsoleMonitorCallback,
    LoggingMonitorCallback,
    CallbackChain,
)

# Configure monitoring
config = MonitorConfig.production()

# Create callback chain
callbacks = CallbackChain([
    ConsoleMonitorCallback(show_progress=True, use_colors=True),
    LoggingMonitorCallback(logger_name="profiler.monitor"),
])

# Create monitor
monitor = DistributedMonitor(config, callbacks=[callbacks])

# Create profiler with monitoring
profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=SparkConfig(master="spark://master:7077"),
    monitor=monitor,
)

# Profile with full observability
profile = profiler.profile("hdfs://data/large.parquet")

# Get monitoring statistics
stats = monitor.get_statistics()
print(f"Tasks completed: {stats['tasks_completed']}")
print(f"Average duration: {stats['avg_task_duration']:.2f}s")
print(f"Success rate: {stats['success_rate']:.1%}")
```

## CLI Usage

```bash
# Local parallel processing
th profile data.csv --distributed --backend local --workers 4

# Spark processing
th profile hdfs://data/large.parquet --distributed --backend spark \
  --master spark://master:7077

# Dask processing
th profile s3://bucket/data/*.parquet --distributed --backend dask \
  --scheduler distributed

# Automatic backend selection
th profile data.parquet --distributed --backend auto
```

## 환경 변수

| 실무 운영 가이드에서 Variable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|---------|
| 실무 운영 가이드에서 `TRUTHOUND_DISTRIBUTED_BACKEND`, TRUTHOUND_DISTRIBUTED_BACKEND을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `local`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TRUTHOUND_NUM_WORKERS`, TRUTHOUND_NUM_WORKERS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 CPU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SPARK_HOME`, SPARK_HOME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DASK_SCHEDULER_ADDRESS`, DASK_SCHEDULER_ADDRESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RAY_ADDRESS`, RAY_ADDRESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ray을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 통합 Example

```python
from truthound.profiler.distributed import (
    DistributedProfiler,
    DistributedBackend,
    SparkConfig,
    PartitionStrategy,
)

# Profile large data on Spark cluster
config = SparkConfig(
    master="spark://master:7077",
    executor_memory="16g",
    num_executors=20,
    spark_config={
        "spark.sql.shuffle.partitions": "200",
    },
)

profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=config,
    partition_strategy=PartitionStrategy.HYBRID,
)

# Profile 100GB data
profile = profiler.profile("hdfs://data/100gb_dataset.parquet")

print(f"Rows: {profile.row_count:,}")
print(f"Columns: {profile.column_count}")
print(f"Duration: {profile.profile_duration_ms / 1000:.2f}s")
```

## 성능 Tips

| 실무 운영 가이드에서 Data, Size을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Recommended, Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Recommended, Settings을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|--------------------|--------------------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Local/Dask을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Process을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dask/Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cluster을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Large을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 다음 단계

- 실무 운영 가이드에서 Sampling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [캐싱](caching.md) - Distributed 캐시 sharing
