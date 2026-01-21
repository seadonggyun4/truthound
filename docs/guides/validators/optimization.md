# 검증기 최적화

Truthound의 최적화 모듈은 대규모 데이터셋에서 검증기 실행 성능을 극대화하기 위한 도구들을 제공합니다.

## 개요

최적화 모듈은 다음 영역을 다룹니다:

| 영역 | 모듈 | 주요 기능 |
|-----|------|---------|
| **DAG 실행** | `orchestrator` | 의존성 기반 실행 순서, 병렬 실행 |
| **그래프 알고리즘** | `graph` | 사이클 감지, 위상 정렬, DFS |
| **공분산 계산** | `covariance` | 증분 공분산, Woodbury 업데이트 |
| **지리 연산** | `geo` | 벡터화된 Haversine, 공간 인덱싱 |
| **집계 최적화** | `aggregation` | 지연 집계, 스트리밍 조인 |
| **프로파일링** | `profiling` | 실행 시간, 메모리, 처리량 측정 |

## DAG 기반 실행 오케스트레이션

`ValidatorDAG`는 검증기 간의 의존성을 관리하고 최적의 실행 순서를 결정합니다.

### 기본 사용법

```python
from truthound.validators.optimization import (
    ValidatorDAG,
    ExecutionPlan,
    ParallelExecutionStrategy,
)

# DAG 생성
dag = ValidatorDAG()

# 검증기 추가
dag.add_validators([
    NullValidator(),
    DuplicateValidator(),
    RangeValidator(),
])

# 실행 계획 생성
plan = dag.build_execution_plan()

# 병렬 전략으로 실행
strategy = ParallelExecutionStrategy(max_workers=4)
results = plan.execute(lf, strategy)
```

### 실행 위상 (ValidatorPhase)

검증기는 자동으로 실행 위상에 할당됩니다:

| 위상 | 설명 | 우선순위 |
|-----|------|---------|
| `SCHEMA` | 스키마 검증 (컬럼 존재, 타입) | 1 |
| `COMPLETENESS` | Null 검사, 누락 값 | 2 |
| `UNIQUENESS` | 중복 감지, 키 검증 | 3 |
| `FORMAT` | 패턴 매칭, 형식 검증 | 4 |
| `RANGE` | 값 범위, 분포 검사 | 5 |
| `STATISTICAL` | 통계, 이상치 | 6 |
| `CROSS_TABLE` | 다중 테이블 검증 | 7 |
| `CUSTOM` | 사용자 정의 | 8 |

### ValidatorNode

검증기를 DAG에 추가할 때 메타데이터를 지정할 수 있습니다:

```python
from truthound.validators.optimization import ValidatorNode, ValidatorPhase

node = dag.add_validator(
    validator=MyValidator(),
    dependencies={"null_validator"},  # 명시적 의존성
    provides={"completeness"},         # 제공하는 기능
    phase=ValidatorPhase.STATISTICAL,  # 실행 위상
    priority=50,                       # 위상 내 우선순위 (낮을수록 먼저)
    estimated_cost=2.5,                # 예상 비용 (적응형 스케줄링용)
)
```

### 실행 전략

#### SequentialExecutionStrategy

검증기를 순차적으로 실행합니다:

```python
from truthound.validators.optimization import SequentialExecutionStrategy

strategy = SequentialExecutionStrategy()
result = plan.execute(lf, strategy)
```

#### ParallelExecutionStrategy

`ThreadPoolExecutor`를 사용하여 병렬 실행:

```python
from truthound.validators.optimization import ParallelExecutionStrategy

strategy = ParallelExecutionStrategy(max_workers=8)
result = plan.execute(lf, strategy)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `max_workers` | `int \| None` | `None` | 최대 워커 수. None이면 `min(32, cpu_count + 4)` |

#### AdaptiveExecutionStrategy

레벨의 검증기 수에 따라 순차/병렬 전략을 동적으로 선택:

```python
from truthound.validators.optimization import AdaptiveExecutionStrategy

strategy = AdaptiveExecutionStrategy(
    parallel_threshold=3,  # 3개 이상일 때 병렬 실행
    max_workers=4,
)
result = plan.execute(lf, strategy)
```

### ExecutionResult

실행 결과는 다양한 메트릭을 제공합니다:

```python
result = plan.execute(lf, strategy)

# 기본 메트릭
print(f"총 소요 시간: {result.total_duration_ms}ms")
print(f"총 검증기: {result.total_validators}")
print(f"성공: {result.success_count}")
print(f"실패: {result.failure_count}")
print(f"전체 이슈: {len(result.all_issues)}")

# 상세 메트릭
metrics = result.get_metrics()
# {
#     "total_duration_ms": 150.5,
#     "total_validators": 10,
#     "total_issues": 25,
#     "success_count": 8,
#     "failure_count": 2,
#     "levels": 4,
#     "strategy": "parallel",
#     "parallelism_factor": 2.5,  # 순차 시간 / 실제 시간
# }
```

### 편의 함수

```python
from truthound.validators.optimization import (
    create_execution_plan,
    execute_validators,
)

# 실행 계획 생성
plan = create_execution_plan(
    validators=[v1, v2, v3],
    dependencies={"v2": {"v1"}},  # v2는 v1에 의존
)

# 한 번에 실행
result = execute_validators(
    validators=[v1, v2, v3],
    lf=df.lazy(),
    strategy=ParallelExecutionStrategy(max_workers=4),
)
```

## 그래프 순회 알고리즘

계층 구조 및 관계 데이터 검증을 위한 최적화된 그래프 알고리즘입니다.

### IterativeDFS

재귀 제한을 피하기 위한 반복적 깊이 우선 탐색:

```python
from truthound.validators.optimization import IterativeDFS

adjacency = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}

dfs = IterativeDFS(adjacency)

# 전위 순회
for node in dfs.traverse(start="A", order="preorder"):
    print(node)  # A, B, D, C

# 경로 찾기
path = dfs.find_path("A", "D")  # ["A", "B", "D"]

# 깊이 계산
depths = dfs.compute_depths(roots=["A"])  # {"A": 0, "B": 1, "C": 1, "D": 2}
```

### TarjanSCC

Tarjan의 강연결 요소(SCC) 알고리즘으로 O(V+E) 시간에 사이클 감지:

```python
from truthound.validators.optimization import TarjanSCC, CycleInfo

adjacency = {
    "A": ["B"],
    "B": ["C"],
    "C": ["A", "D"],  # A-B-C 사이클
    "D": ["D"],       # 자기 루프
}

tarjan = TarjanSCC(adjacency)

# 모든 SCC 찾기
sccs = tarjan.find_sccs()
# [["A", "C", "B"], ["D"]]  # 크기 > 1인 SCC는 사이클

# 사이클 정보 가져오기
cycles = tarjan.find_cycles()
for cycle in cycles:
    print(f"사이클: {cycle.nodes}")
    print(f"길이: {cycle.length}")
    print(f"자기 루프: {cycle.is_self_loop}")
```

### TopologicalSort

Kahn의 알고리즘을 사용한 위상 정렬:

```python
from truthound.validators.optimization import TopologicalSort

dependencies = {
    "task_a": ["task_b", "task_c"],  # a 다음에 b, c
    "task_b": ["task_d"],
    "task_c": ["task_d"],
    "task_d": [],
}

sorter = TopologicalSort(dependencies)

# 정렬 수행
try:
    order = sorter.sort()  # ["task_a", "task_b", "task_c", "task_d"]
except ValueError:
    print("사이클이 존재합니다")

# 사이클 확인
if sorter.has_cycles():
    print("사이클 감지")
```

### GraphTraversalMixin

검증기에서 그래프 연산을 위한 믹스인:

```python
from truthound.validators.optimization import GraphTraversalMixin

class HierarchyValidator(BaseValidator, GraphTraversalMixin):
    def validate(self, df):
        # 인접 리스트 생성
        adj = self.build_adjacency_list(
            df,
            id_column="id",
            parent_column="parent_id",
            cache_key="my_hierarchy",  # 캐싱 키
        )

        # 모든 사이클 찾기
        cycles = self.find_all_cycles(adj)

        # 계층 사이클 찾기 (부모-자식 관계)
        child_to_parent = self.build_child_to_parent(df, "id", "parent_id")
        hierarchy_cycles = self.find_hierarchy_cycles(
            child_to_parent,
            max_depth=1000,
        )

        # 노드 깊이 계산
        depths = self.compute_node_depths(adj, roots=["root"])

        # 위상 정렬
        sorted_nodes = self.topological_sort(adj)

        # 캐시 정리
        self.clear_adjacency_cache()
```

## 공분산 계산 최적화

대규모 데이터셋에서 마할라노비스 거리 등 다변량 메서드를 위한 효율적인 공분산 계산입니다.

### IncrementalCovariance

Welford의 온라인 알고리즘을 사용한 증분 공분산:

```python
from truthound.validators.optimization import IncrementalCovariance

# 10개 특성에 대한 증분 공분산
cov = IncrementalCovariance(n_features=10)

# 스트리밍 데이터 업데이트
for batch in data_stream:
    cov.update_batch(batch)  # 배치 업데이트

# 단일 샘플 업데이트
cov.update(sample)

# 결과 가져오기
result = cov.get_result(regularization=1e-6)
print(result.mean)       # 평균 벡터
print(result.covariance) # 공분산 행렬
print(result.n_samples)  # 샘플 수
```

### WoodburyCovariance

효율적인 rank-k 업데이트를 위한 Woodbury 행렬 항등식:

```python
from truthound.validators.optimization import WoodburyCovariance

# 데이터에서 생성
cov = WoodburyCovariance.from_data(
    training_data,
    regularization=1e-6,
)

# 효율적인 샘플 추가/제거
cov.add_sample(new_sample)
cov.remove_sample(old_sample)

# 마할라노비스 거리 계산
distance = cov.mahalanobis(query_point)
distances = cov.mahalanobis_batch(query_points)
```

### RobustCovarianceEstimator

이상치에 강건한 MCD(Minimum Covariance Determinant) 추정:

```python
from truthound.validators.optimization import RobustCovarianceEstimator

estimator = RobustCovarianceEstimator(
    contamination=0.1,    # 예상 이상치 비율
    n_subsamples=10,      # 서브샘플 수
    subsample_size=500,   # 각 서브샘플 크기
    random_state=42,
)

result = estimator.fit(large_data)
# result.is_robust = True
```

### BatchCovarianceMixin

검증기에서 배치 공분산 계산을 위한 믹스인:

```python
from truthound.validators.optimization import BatchCovarianceMixin

class MahalanobisValidator(BaseValidator, BatchCovarianceMixin):
    def validate(self, data):
        # 자동으로 최적의 방법 선택
        result = self.compute_covariance_auto(
            data,
            use_robust=True,
        )

        # 마할라노비스 거리 계산
        distances = self.compute_mahalanobis_distances(data, result)

        # Woodbury 공분산 생성 (효율적 업데이트용)
        woodbury = self.create_woodbury_covariance(data)
```

**설정 속성:**

| 속성 | 기본값 | 설명 |
|-----|--------|------|
| `_batch_size` | 10000 | 증분 처리 배치 크기 |
| `_robust_threshold` | 5000 | 이 값 이상이면 서브샘플링된 강건 추정 |
| `_regularization` | 1e-6 | 정칙화 파라미터 |

## 지리 연산 최적화

벡터화된 지리 공간 계산으로 거리 및 폴리곤 검증 성능을 향상시킵니다.

### VectorizedGeoMixin

```python
from truthound.validators.optimization import (
    VectorizedGeoMixin,
    DistanceUnit,
    BoundingBox,
)

class GeoValidator(BaseValidator, VectorizedGeoMixin):
    def validate(self, coords):
        lats = coords[:, 0]
        lons = coords[:, 1]

        # 벡터화된 Haversine 거리
        distances = self.haversine_vectorized(
            lat1=lats,
            lon1=lons,
            lat2=target_lat,
            lon2=target_lon,
            unit=DistanceUnit.KILOMETERS,
        )

        # 쌍별 거리 행렬
        dist_matrix = self.pairwise_distances(
            lats1, lons1, lats2, lons2,
            unit=DistanceUnit.METERS,
        )

        # 청크 처리로 메모리 효율적 계산
        for start, end, chunk in self.pairwise_distances_chunked(
            lats1, lons1, lats2, lons2,
            chunk_size=50000,
        ):
            process_chunk(chunk)

        # 최근접 점 찾기
        idx, dist = self.nearest_point(
            query_lat, query_lon, lats, lons,
        )

        # k-최근접 이웃
        indices, distances = self.k_nearest_points(
            query_lat, query_lon, lats, lons, k=5,
        )

        # 반경 내 점 찾기
        indices, distances = self.points_within_radius(
            center_lat, center_lon, lats, lons,
            radius=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # 방위각 계산
        bearings = self.bearing_vectorized(lat1, lon1, lat2, lon2)

        # 목적지 점 계산
        dest_lat, dest_lon = self.destination_point(
            lat, lon,
            bearing=45,
            distance=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # 바운딩 박스 생성
        bbox = self.create_bounding_box(
            center_lat, center_lon,
            radius=50,
            unit=DistanceUnit.KILOMETERS,
        )

        # 바운딩 박스로 필터링
        mask, filtered_lats, filtered_lons = self.filter_by_bounding_box(
            lats, lons, bbox,
        )

        # 좌표 유효성 검사
        valid_mask = self.validate_coordinates(lats, lons)
```

### DistanceUnit

지원하는 거리 단위:

| 단위 | 지구 반경 |
|-----|----------|
| `METERS` | 6,371,000 m |
| `KILOMETERS` | 6,371 km |
| `MILES` | 3,958.8 mi |
| `NAUTICAL_MILES` | 3,440.1 nm |

### SpatialIndexMixin

BallTree를 사용한 효율적인 공간 쿼리:

```python
from truthound.validators.optimization import SpatialIndexMixin

class IndexedGeoValidator(BaseValidator, SpatialIndexMixin):
    def setup(self, reference_coords):
        # 공간 인덱스 구축
        self.build_spatial_index(
            lats=reference_coords[:, 0],
            lons=reference_coords[:, 1],
            leaf_size=40,
        )

    def validate(self, query_coords):
        # 반경 내 쿼리
        results = self.query_radius(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            radius_km=10,
        )

        # k-최근접 이웃 쿼리
        distances_km, indices = self.query_nearest(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            k=5,
        )

        # 인덱스 정리
        self.clear_spatial_index()
```

## 집계 최적화

Polars의 지연 평가를 활용하여 메모리 효율적인 집계 연산을 수행합니다.

### LazyAggregationMixin

```python
from truthound.validators.optimization import LazyAggregationMixin
import polars as pl

class CrossTableValidator(BaseValidator, LazyAggregationMixin):
    def validate(self, orders, order_items):
        # 지연 집계
        result = self.aggregate_lazy(
            lf=order_items.lazy(),
            group_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("total_qty"),
                pl.col("price").sum().alias("total_price"),
            ],
            cache_key="order_totals",  # 캐싱
        )

        # 조인과 집계를 한 번에
        result = self.aggregate_with_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            left_on="order_id",
            right_on="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            how="left",
        )

        # 대용량 테이블을 위한 스트리밍 조인
        result = self.streaming_aggregate_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            join_key="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            slice_size=100000,
        )

        # 집계 값 비교
        mismatches = self.compare_aggregates(
            source=orders,
            aggregated=result,
            key_column="order_id",
            source_column="expected_total",
            agg_column="total_qty",
            tolerance=0.01,
        )

        # 증분 집계
        updated = self.incremental_aggregate(
            existing=previous_result,
            new_data=new_items.lazy(),
            group_by="order_id",
            sum_columns=["quantity", "price"],
            count_column="item_count",
        )

        # 윈도우 집계
        with_windows = self.window_aggregate(
            lf=order_items.lazy(),
            partition_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("order_total"),
                pl.col("price").mean().alias("order_avg_price"),
            ],
        )

        # 세미 조인 필터
        filtered = self.semi_join_filter(
            main=orders.lazy(),
            filter_by=active_orders.lazy(),
            on="order_id",
            anti=False,  # True면 anti-join
        )

        # 다중 테이블 집계
        result = self.multi_table_aggregate(
            tables={
                "orders": orders.lazy(),
                "items": order_items.lazy(),
                "products": products.lazy(),
            },
            joins=[
                ("orders", "items", ["order_id"]),
                ("items", "products", ["product_id"]),
            ],
            final_agg=[
                pl.col("quantity").sum(),
                pl.col("unit_price").mean(),
            ],
            final_group_by="category",
        )

        # 캐시 정리
        self.clear_aggregation_cache()
```

### AggregationExpressionBuilder

플루언트 인터페이스로 집계 표현식 생성:

```python
from truthound.validators.optimization import AggregationExpressionBuilder

builder = AggregationExpressionBuilder()

exprs = (
    builder
    .sum("quantity", alias="total_qty")
    .mean("price", alias="avg_price")
    .min("created_at", alias="first_order")
    .max("created_at", alias="last_order")
    .std("price", alias="price_std")
    .count(alias="order_count")
    .n_unique("customer_id", alias="unique_customers")
    .first("status")
    .last("updated_at")
    .custom(pl.col("discount").filter(pl.col("discount") > 0).mean())
    .build()
)

# 사용
result = lf.group_by("category").agg(exprs)
```

## 검증기 프로파일링

검증기 실행 성능을 측정하고 분석합니다.

### ProfilerConfig

프로파일러 설정:

```python
from truthound.validators.optimization import (
    ProfilerConfig,
    ProfilerMode,
)

# 기본 설정 (타이밍만)
config = ProfilerConfig.basic()

# 표준 설정 (타이밍 + 메모리)
config = ProfilerConfig()  # 기본값

# 상세 설정 (스냅샷 포함)
config = ProfilerConfig.detailed()

# 진단 설정 (최대 상세)
config = ProfilerConfig.diagnostic()

# 사용자 정의 설정
config = ProfilerConfig(
    mode=ProfilerMode.STANDARD,
    track_memory=True,
    track_gc=True,
    track_throughput=True,
    record_snapshots=False,
    max_snapshots=1000,
    memory_warning_mb=100,
    memory_critical_mb=500,
)
```

### ValidatorProfiler

```python
from truthound.validators.optimization import (
    ValidatorProfiler,
    ProfilerConfig,
)

# 프로파일러 생성
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())

# 세션 시작
profiler.start_session("validation_run_1", attributes={"env": "prod"})

# 검증기 프로파일링
for validator in validators:
    with profiler.profile(validator, rows_processed=100000) as ctx:
        issues = validator.validate(lf)
        ctx.set_issue_count(len(issues))
        ctx.add_attribute("columns", ["a", "b", "c"])

# 세션 종료
session = profiler.end_session()

# 결과 분석
print(session.summary())
print(session.to_json())

# 가장 느린 검증기
slowest = profiler.get_slowest_validators(n=10)
# [("SlowValidator", 150.5), ("AnotherValidator", 100.2), ...]

# 메모리 집약적 검증기
memory_heavy = profiler.get_memory_intensive_validators(n=10)

# 전체 요약
summary = profiler.summary()
# {
#     "total_validators": 15,
#     "total_executions": 150,
#     "total_issues": 2500,
#     "total_time_ms": 5000.5,
#     "completed_sessions": 3,
#     "current_session_active": False,
#     "memory_tracking_available": True,
# }
```

### 편의 함수

```python
from truthound.validators.optimization import (
    profile_validator,
    profiled,
    get_default_profiler,
)

# 컨텍스트 매니저
with profile_validator(my_validator, rows_processed=10000) as ctx:
    issues = my_validator.validate(lf)
    ctx.set_issue_count(len(issues))

print(ctx.metrics.timing.mean_ms)

# 데코레이터
class MyValidator(Validator):
    @profiled(track_issues=True)
    def validate(self, lf):
        return [issue1, issue2]

# 전역 프로파일러
profiler = get_default_profiler()
```

### ValidatorMetrics

수집되는 메트릭:

```python
metrics = profiler.get_metrics("MyValidator")

# 타이밍 메트릭
print(metrics.timing.mean_ms)    # 평균 실행 시간
print(metrics.timing.median_ms)  # 중앙값
print(metrics.timing.p95_ms)     # 95 백분위수
print(metrics.timing.p99_ms)     # 99 백분위수
print(metrics.timing.std_ms)     # 표준 편차

# 메모리 메트릭
print(metrics.memory.mean_peak_mb)       # 평균 피크 메모리
print(metrics.memory.max_peak_mb)        # 최대 피크 메모리
print(metrics.memory.total_gc_collections)  # GC 수집 횟수

# 처리량 메트릭
print(metrics.throughput.mean_rows_per_sec)  # 초당 처리 행 수
print(metrics.throughput.total_rows)         # 총 처리 행 수

# 이슈 메트릭
print(metrics.total_issues)  # 총 발견된 이슈
print(metrics.mean_issues)   # 평균 이슈 수
print(metrics.error_counts)  # 오류 횟수
```

### ProfilingReport

보고서 생성:

```python
from truthound.validators.optimization import ProfilingReport

report = ProfilingReport(profiler)

# 텍스트 요약
print(report.text_summary())
# ============================================================
# VALIDATOR PROFILING REPORT
# ============================================================
# Total Validators: 15
# Total Executions: 150
# Total Issues Found: 2500
# Total Time: 5000.50ms
#
# ------------------------------------------------------------
# TOP 10 SLOWEST VALIDATORS (by mean execution time)
# ------------------------------------------------------------
#  1. SlowValidator: 150.50ms
#  2. AnotherValidator: 100.20ms
# ...

# HTML 보고서
html = report.html_report()
with open("profile_report.html", "w") as f:
    f.write(html)

# Prometheus 형식 내보내기
prometheus_metrics = profiler.to_prometheus()
# # HELP validator_execution_duration_ms Validator execution duration
# # TYPE validator_execution_duration_ms gauge
# validator_execution_duration_ms_mean{validator="MyValidator",category="completeness"} 150.500
# ...
```

## 통합 예제

모든 최적화 기능을 함께 사용하는 예제:

```python
import polars as pl
from truthound.validators.optimization import (
    # DAG 오케스트레이션
    ValidatorDAG,
    ParallelExecutionStrategy,
    # 프로파일링
    ValidatorProfiler,
    ProfilerConfig,
    # 믹스인
    GraphTraversalMixin,
    BatchCovarianceMixin,
    VectorizedGeoMixin,
    LazyAggregationMixin,
)
from truthound.validators import NullValidator, RangeValidator


# 커스텀 검증기 정의
class OptimizedHierarchyValidator(
    BaseValidator,
    GraphTraversalMixin,
    BatchCovarianceMixin,
):
    def validate(self, df):
        # 그래프 사이클 감지
        adj = self.build_adjacency_list(df, "id", "parent_id")
        cycles = self.find_all_cycles(adj)

        # 이상치 감지를 위한 공분산
        numeric_data = df.select(pl.col(pl.Float64)).to_numpy()
        cov_result = self.compute_covariance_auto(numeric_data, use_robust=True)
        distances = self.compute_mahalanobis_distances(numeric_data, cov_result)

        return []


# 프로파일러 설정
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())
profiler.start_session("optimized_validation")

# DAG 구성
dag = ValidatorDAG()
dag.add_validators([
    NullValidator(),
    RangeValidator(min_value=0),
    OptimizedHierarchyValidator(),
])

# 실행 계획 생성 및 실행
plan = dag.build_execution_plan()
print(plan.get_summary())

# 병렬 실행
strategy = ParallelExecutionStrategy(max_workers=4)

df = pl.DataFrame({"id": [1, 2, 3], "parent_id": [None, 1, 2], "value": [10, 20, 30]})

with profiler.profile(plan, rows_processed=len(df)) as ctx:
    result = plan.execute(df.lazy(), strategy)
    ctx.set_issue_count(len(result.all_issues))

# 세션 종료 및 보고서
session = profiler.end_session()
print(session.to_json())
```

## API 레퍼런스

### orchestrator 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `ValidatorDAG` | 검증기 의존성 DAG |
| `ValidatorNode` | 검증기 노드 래퍼 |
| `ValidatorPhase` | 실행 위상 열거형 |
| `ExecutionPlan` | 실행 계획 |
| `ExecutionLevel` | 병렬 실행 가능한 검증기 그룹 |
| `ExecutionResult` | 실행 결과 |
| `ExecutionStrategy` | 실행 전략 추상 클래스 |
| `SequentialExecutionStrategy` | 순차 실행 전략 |
| `ParallelExecutionStrategy` | 병렬 실행 전략 |
| `AdaptiveExecutionStrategy` | 적응형 실행 전략 |
| `create_execution_plan()` | 실행 계획 생성 편의 함수 |
| `execute_validators()` | 검증기 실행 편의 함수 |

### graph 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `IterativeDFS` | 반복적 깊이 우선 탐색 |
| `TarjanSCC` | Tarjan SCC 알고리즘 |
| `TopologicalSort` | 위상 정렬 |
| `GraphTraversalMixin` | 그래프 순회 믹스인 |
| `CycleInfo` | 사이클 정보 데이터클래스 |
| `NodeState` | 노드 상태 열거형 |

### covariance 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `IncrementalCovariance` | Welford의 증분 공분산 |
| `WoodburyCovariance` | Woodbury 업데이트 공분산 |
| `RobustCovarianceEstimator` | MCD 기반 강건 추정 |
| `BatchCovarianceMixin` | 배치 공분산 믹스인 |
| `CovarianceResult` | 공분산 결과 데이터클래스 |

### geo 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `VectorizedGeoMixin` | 벡터화된 지리 연산 믹스인 |
| `SpatialIndexMixin` | 공간 인덱싱 믹스인 |
| `DistanceUnit` | 거리 단위 열거형 |
| `BoundingBox` | 바운딩 박스 데이터클래스 |

### aggregation 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `LazyAggregationMixin` | 지연 집계 믹스인 |
| `AggregationExpressionBuilder` | 집계 표현식 빌더 |
| `AggregationResult` | 집계 결과 데이터클래스 |
| `JoinStrategy` | 조인 전략 설정 |

### profiling 모듈

| 클래스/함수 | 설명 |
|-----------|------|
| `ValidatorProfiler` | 메인 프로파일러 클래스 |
| `ProfilerConfig` | 프로파일러 설정 |
| `ProfilerMode` | 프로파일러 모드 열거형 |
| `ValidatorMetrics` | 검증기 메트릭 |
| `TimingMetrics` | 타이밍 메트릭 |
| `MemoryMetrics` | 메모리 메트릭 |
| `ThroughputMetrics` | 처리량 메트릭 |
| `ProfilingSession` | 프로파일링 세션 |
| `ExecutionSnapshot` | 실행 스냅샷 |
| `ProfilingReport` | 보고서 생성기 |
| `profile_validator()` | 프로파일링 컨텍스트 매니저 |
| `profiled()` | 프로파일링 데코레이터 |
