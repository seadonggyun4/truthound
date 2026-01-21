# Distributed Processing

이 문서는 대규모 데이터셋을 위한 분산 프로파일링 시스템을 설명합니다.

## 개요

`src/truthound/profiler/distributed.py`에 구현된 분산 처리 시스템은 Spark, Dask, Ray, Local 백엔드를 지원하며 통합된 인터페이스를 제공합니다.

## DistributedBackend

```python
class DistributedBackend(str, Enum):
    """분산 처리 백엔드"""

    LOCAL = "local"    # 로컬 멀티스레드/멀티프로세스
    SPARK = "spark"    # Apache Spark
    DASK = "dask"      # Dask Distributed
    RAY = "ray"        # Ray
    AUTO = "auto"      # 자동 감지
```

## PartitionStrategy

```python
class PartitionStrategy(str, Enum):
    """파티션 전략"""

    ROW_BASED = "row_based"       # 행 기반 분할
    COLUMN_BASED = "column_based" # 컬럼 기반 분할
    HYBRID = "hybrid"             # 하이브리드 (행 + 컬럼)
    HASH = "hash"                 # 해시 기반 분할
```

## 백엔드별 설정

### SparkConfig

```python
@dataclass
class SparkConfig:
    """Spark 설정"""

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
    """Dask 설정"""

    scheduler: str = "threads"  # threads, processes, synchronous
    num_workers: int | None = None  # None = CPU 코어 수
    memory_limit: str = "4GB"
    dashboard_address: str = ":8787"
```

### RayConfig

```python
@dataclass
class RayConfig:
    """Ray 설정"""

    address: str | None = None  # None = 로컬 클러스터
    num_cpus: int | None = None
    num_gpus: int | None = None
    object_store_memory: int | None = None
```

## DistributedProfiler

통합 분산 프로파일러 인터페이스입니다.

```python
from truthound.profiler.distributed import DistributedProfiler, DistributedBackend

# 자동 백엔드 선택
profiler = DistributedProfiler.create(backend=DistributedBackend.AUTO)

# Spark 백엔드
profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=SparkConfig(master="spark://master:7077"),
)

# Dask 백엔드
profiler = DistributedProfiler.create(
    backend=DistributedBackend.DASK,
    config=DaskConfig(scheduler="distributed"),
)

# 프로파일 실행
profile = profiler.profile("hdfs://data/large.parquet")
```

## LocalBackend

로컬 멀티스레드/멀티프로세스 백엔드입니다.

```python
from truthound.profiler.distributed import LocalBackend

backend = LocalBackend(
    num_workers=4,
    use_processes=True,  # True = 프로세스, False = 스레드
)

# 컬럼별 병렬 프로파일링
results = backend.profile_columns_parallel(lf, columns)
```

## SparkBackend

Apache Spark 백엔드입니다.

```python
from truthound.profiler.distributed import SparkBackend, SparkConfig

config = SparkConfig(
    master="spark://master:7077",
    executor_memory="8g",
    num_executors=10,
)

backend = SparkBackend(config)

# Spark DataFrame 프로파일링
profile = backend.profile(spark_df)

# HDFS 파일 프로파일링
profile = backend.profile_file("hdfs://data/large.parquet")
```

## DaskBackend

Dask Distributed 백엔드입니다.

```python
from truthound.profiler.distributed import DaskBackend, DaskConfig

config = DaskConfig(
    scheduler="distributed",
    num_workers=8,
    memory_limit="8GB",
)

backend = DaskBackend(config)

# Dask DataFrame 프로파일링
profile = backend.profile(dask_df)

# 파티션된 파일 프로파일링
profile = backend.profile_file("s3://bucket/data/*.parquet")
```

## RayBackend

Ray 백엔드입니다.

```python
from truthound.profiler.distributed import RayBackend, RayConfig

config = RayConfig(
    address="ray://cluster:10001",
    num_cpus=16,
)

backend = RayBackend(config)

# Ray Dataset 프로파일링
profile = backend.profile(ray_dataset)
```

## BackendRegistry

플러그인 가능한 백엔드 레지스트리입니다.

```python
from truthound.profiler.distributed import BackendRegistry

# 등록된 백엔드 조회
backends = BackendRegistry.list()  # ["local", "spark", "dask", "ray"]

# 백엔드 인스턴스 생성
backend = BackendRegistry.create("spark", config=spark_config)

# 커스텀 백엔드 등록
@BackendRegistry.register("my_backend")
class MyCustomBackend:
    def profile(self, data):
        # 커스텀 프로파일링 로직
        pass
```

## 파티션 전략

### ROW_BASED - 행 기반

```python
from truthound.profiler.distributed import RowBasedPartitioner

partitioner = RowBasedPartitioner(num_partitions=10)
partitions = partitioner.partition(lf)

# 각 파티션은 전체 행의 1/10
```

### COLUMN_BASED - 컬럼 기반

```python
from truthound.profiler.distributed import ColumnBasedPartitioner

partitioner = ColumnBasedPartitioner()
column_groups = partitioner.partition(lf, num_workers=4)

# 각 워커가 1/4의 컬럼을 처리
```

### HYBRID - 하이브리드

```python
from truthound.profiler.distributed import HybridPartitioner

partitioner = HybridPartitioner(
    row_partitions=4,
    column_partitions=2,
)
# 4 x 2 = 8개 파티션
```

## 자동 백엔드 선택

```python
from truthound.profiler.distributed import auto_select_backend

# 환경에 따라 자동 선택
backend = auto_select_backend()

# 선택 로직:
# 1. Spark 세션 존재 → Spark
# 2. Dask 클라이언트 존재 → Dask
# 3. Ray 초기화됨 → Ray
# 4. 기본 → Local
```

## 결과 병합

분산 처리된 결과를 병합합니다.

```python
from truthound.profiler.distributed import ProfileMerger

merger = ProfileMerger()

# 파티션별 프로파일 병합
partial_profiles = [profile1, profile2, profile3, profile4]
merged = merger.merge(partial_profiles)

# 통계 병합 전략:
# - count: 합계
# - mean: 가중 평균
# - min/max: 전체 min/max
# - quantiles: 근사 병합
```

## 모니터링

```python
from truthound.profiler.distributed import DistributedMonitor

monitor = DistributedMonitor()

# 진행 상황 모니터링
profiler = DistributedProfiler.create(
    backend="spark",
    monitor=monitor,
)

# 콜백 등록
monitor.on_partition_complete(lambda p: print(f"Partition {p} complete"))
monitor.on_progress(lambda pct: print(f"Progress: {pct}%"))

profile = profiler.profile(data)
```

## CLI 사용법

```bash
# 로컬 병렬 처리
th profile data.csv --distributed --backend local --workers 4

# Spark 처리
th profile hdfs://data/large.parquet --distributed --backend spark \
  --master spark://master:7077

# Dask 처리
th profile s3://bucket/data/*.parquet --distributed --backend dask \
  --scheduler distributed

# 자동 백엔드 선택
th profile data.parquet --distributed --backend auto
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `TRUTHOUND_DISTRIBUTED_BACKEND` | 기본 백엔드 | `local` |
| `TRUTHOUND_NUM_WORKERS` | 워커 수 | CPU 코어 수 |
| `SPARK_HOME` | Spark 홈 | - |
| `DASK_SCHEDULER_ADDRESS` | Dask 스케줄러 | - |
| `RAY_ADDRESS` | Ray 클러스터 | - |

## 통합 예제

```python
from truthound.profiler.distributed import (
    DistributedProfiler,
    DistributedBackend,
    SparkConfig,
    PartitionStrategy,
)

# Spark 클러스터에서 대규모 데이터 프로파일링
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

# 100GB 데이터 프로파일링
profile = profiler.profile("hdfs://data/100gb_dataset.parquet")

print(f"Rows: {profile.row_count:,}")
print(f"Columns: {profile.column_count}")
print(f"Duration: {profile.profile_duration_ms / 1000:.2f}s")
```

## 성능 팁

| 데이터 크기 | 권장 백엔드 | 권장 설정 |
|------------|------------|-----------|
| < 1GB | Local | 4-8 스레드 |
| 1-10GB | Local/Dask | 프로세스 모드 |
| 10-100GB | Dask/Spark | 클러스터 |
| > 100GB | Spark | 대형 클러스터 |

## 다음 단계

- [샘플링](sampling.md) - 분산 환경에서 샘플링
- [캐싱](caching.md) - 분산 캐시 공유
