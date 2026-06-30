# 검증기 Optimization

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Area을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 주요 기능 |
|------|--------|--------------|
| 실무 운영 가이드에서 DAG, Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `orchestrator`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dependency-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Graph, Algorithms을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `graph`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cycle, DFS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Metric, Deduplication을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `metrics`, VE-3을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `SharedMetricStore`, Cross-validator, SharedMetricStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Conditional, Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `base`, VE-4을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `SkipCondition`, SkipCondition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Tier, Fallback을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `base`, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Batch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Covariance, Computation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `covariance`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Incremental, Woodbury을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Geographic, Operations을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `geo`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Vectorized, Haversine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Aggregation, Optimization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `aggregation`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Lazy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| **프로파일링** | 실무 운영 가이드에서 `profiling`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## DAG-Based Execution 오케스트레이션

실무 운영 가이드에서 `ValidatorDAG`, ValidatorDAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.validators.optimization import (
    ValidatorDAG,
    ExecutionPlan,
    ParallelExecutionStrategy,
)

# Create DAG
dag = ValidatorDAG()

# Add validators
dag.add_validators([
    NullValidator(),
    DuplicateValidator(),
    RangeValidator(),
])

# Build execution plan
plan = dag.build_execution_plan()

# Execute with parallel strategy
strategy = ParallelExecutionStrategy(max_workers=4)
results = plan.execute(lf, strategy)
```

### Execution Phases (ValidatorPhase)

실무 운영 가이드에서 Validators을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Priority을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|----------|
| 실무 운영 가이드에서 `SCHEMA`, SCHEMA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 검증 (컬럼 existence, types) | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COMPLETENESS`, COMPLETENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `UNIQUENESS`, UNIQUENESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Duplicate detection, key 검증 | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FORMAT`, FORMAT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Pattern matching, format 검증 | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RANGE`, RANGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Value을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STATISTICAL`, STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Statistics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CROSS_TABLE`, CROSS_TABLE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Multi-테이블 검증 | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CUSTOM`, CUSTOM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 User-defined을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### ValidatorNode

실무 운영 가이드에서 Metadata, DAG을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import ValidatorNode, ValidatorPhase

node = dag.add_validator(
    validator=MyValidator(),
    dependencies={"null_validator"},  # Explicit dependencies
    provides={"completeness"},         # Provided capabilities
    phase=ValidatorPhase.STATISTICAL,  # Execution phase
    priority=50,                       # Priority within phase (lower = earlier)
    estimated_cost=2.5,                # Estimated cost (for adaptive scheduling)
)
```

### Execution Strategies

#### SequentialExecutionStrategy

Executes 검증기 sequentially:

```python
from truthound.validators.optimization import SequentialExecutionStrategy

strategy = SequentialExecutionStrategy()
result = plan.execute(lf, strategy)
```

#### ParallelExecutionStrategy

실무 운영 가이드에서 `ThreadPoolExecutor`, Parallel, ThreadPoolExecutor을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import ParallelExecutionStrategy

strategy = ParallelExecutionStrategy(max_workers=8)
result = plan.execute(lf, strategy)
```

실무 운영 가이드에서 Parameters을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `max_workers`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `min(32, cpu_count + 4)`, Maximum, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

#### AdaptiveExecutionStrategy

실무 운영 가이드에서 Dynamically을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import AdaptiveExecutionStrategy

strategy = AdaptiveExecutionStrategy(
    parallel_threshold=3,  # Use parallel when 3 or more validators
    max_workers=4,
)
result = plan.execute(lf, strategy)
```

### ExecutionResult

실무 운영 가이드에서 Execution을(를) 다루는 항목입니다:

```python
result = plan.execute(lf, strategy)

# Basic metrics
print(f"Total duration: {result.total_duration_ms}ms")
print(f"Total validators: {result.total_validators}")
print(f"Success: {result.success_count}")
print(f"Failure: {result.failure_count}")
print(f"Total issues: {len(result.all_issues)}")

# Detailed metrics
metrics = result.get_metrics()
# {
#     "total_duration_ms": 150.5,
#     "total_validators": 10,
#     "total_issues": 25,
#     "success_count": 8,
#     "failure_count": 2,
#     "levels": 4,
#     "strategy": "parallel",
#     "parallelism_factor": 2.5,  # Sequential time / actual time
# }
```

### Convenience Functions

```python
from truthound.validators.optimization import (
    create_execution_plan,
    execute_validators,
)

# Create execution plan
plan = create_execution_plan(
    validators=[v1, v2, v3],
    dependencies={"v2": {"v1"}},  # v2 depends on v1
)

# Execute at once
result = execute_validators(
    validators=[v1, v2, v3],
    lf=df.lazy(),
    strategy=ParallelExecutionStrategy(max_workers=4),
)
```

## Metric Deduplication (VE-3)

실무 운영 가이드에서 `null_count`, `email`, `SharedMetricStore`, SharedMetricStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 아키텍처

```
ExpressionBatchExecutor._precompute_shared_metrics()
├── Collects MetricKey lists from all validators via get_required_metrics()
├── Deduplicates into unique MetricKey set
├── Builds single lf.select([expr1, expr2, ...]).collect()
└── Stores results in SharedMetricStore
```

### Declaring Metric Dependencies

```python
from truthound.validators.base import Validator
from truthound.validators.metrics import MetricKey, CommonMetrics

class MyValidator(Validator):
    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        keys = []
        for col in columns:
            key, _ = CommonMetrics.null_count(col)
            keys.append(key)
            key, _ = CommonMetrics.row_count()
            keys.append(key)
        return keys

    def validate_with_metrics(self, lf, metric_store):
        for col in self._get_target_columns(lf):
            key, _ = CommonMetrics.null_count(col)
            null_count = metric_store.get(key)
            # Use cached value instead of recomputing
```

### Available Common 메트릭

| 실무 운영 가이드에서 Metric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Scope을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Expression을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|------------|
| 실무 운영 가이드에서 `row_count()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 테이블 | 실무 운영 가이드에서 `pl.len()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `null_count(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).is_null().sum()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `non_null_count(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).is_not_null().sum()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `n_unique(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).n_unique()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `mean(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).mean()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `std(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).std()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `min(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).min()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).max()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sum(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).sum()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `quantile(col, q)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).quantile(q)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `median(col)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 | 실무 운영 가이드에서 `pl.col(col).median()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SharedMetricStore API

```python
from truthound.validators.metrics import SharedMetricStore, MetricKey

store = SharedMetricStore()

# Basic operations
store.put(key, value)
value = store.get(key)             # Returns None if missing
value = store.get_or_compute(key, compute_fn)

# Bulk operations
store.put_many({key1: val1, key2: val2})
results = store.get_many([key1, key2])
missing = store.missing_keys([key1, key2, key3])

# Statistics
stats = store.stats  # MetricStoreStats(hits, misses, size)
```

## 실무 운영 가이드 개요

실무 운영 가이드에서 Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Priority Hierarchy

실무 운영 가이드에서 Validators을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Priority, Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 예시 |
|----------------|----------|----------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 스키마 | 실무 운영 가이드에서 ColumnExistsValidator, ColumnTypeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Completeness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 NullValidator, NotNullValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Uniqueness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 UniqueValidator, DuplicateValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Distribution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 RangeValidator, BetweenValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Referential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ForeignKeyValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Declaring Skip Conditions

```python
from truthound.validators.base import Validator, SkipCondition

class DistributionValidator(Validator):
    dependencies = {"schema_check", "null_check"}
    priority = 75

    def get_skip_conditions(self) -> list[SkipCondition]:
        return [
            # Skip if schema validation failed entirely
            SkipCondition(depends_on="schema_check", skip_when="failed"),
            # Skip if null check found critical-level nulls
            SkipCondition(depends_on="null_check", skip_when="critical"),
        ]
```

### 3-Stage Dependency Resolution

1. 실무 운영 가이드에서 `dependencies`, Dependency, FAILED/TIMEOUT/SKIPPED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 `ValidatorExecutionResult`, SkipCondition, ValidatorExecutionResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `_filter_columns_by_context()`, Column을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 3-Tier Expression Fallback (VE-5)

실무 운영 가이드에서 `ExpressionBatchExecutor`, ExpressionBatchExecutor을(를) 다루는 항목입니다:

```
Tier 1: Batch all validators → single lf.select([...]).collect()
        │ ComputeError / SchemaError
        ▼
Tier 2: Per-validator execution → individual collect() per validator
        │ failure on specific validator
        ▼
Tier 3: Per-expression execution → individual collect() per expression
        partial_failure_mode controls behavior:
          "collect" → gather partial results, continue
          "skip"    → discard failing expressions
          "raise"   → re-raise the exception
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Graph Traversal Algorithms

실무 운영 가이드에서 Optimized을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### IterativeDFS

실무 운영 가이드에서 Iterative을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import IterativeDFS

adjacency = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}

dfs = IterativeDFS(adjacency)

# Preorder traversal
for node in dfs.traverse(start="A", order="preorder"):
    print(node)  # A, B, D, C

# Find path
path = dfs.find_path("A", "D")  # ["A", "B", "D"]

# Compute depths
depths = dfs.compute_depths(roots=["A"])  # {"A": 0, "B": 1, "C": 1, "D": 2}
```

### TarjanSCC

실무 운영 가이드에서 Tarjan, Strongly, Connected, Components을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import TarjanSCC, CycleInfo

adjacency = {
    "A": ["B"],
    "B": ["C"],
    "C": ["A", "D"],  # A-B-C cycle
    "D": ["D"],       # Self-loop
}

tarjan = TarjanSCC(adjacency)

# Find all SCCs
sccs = tarjan.find_sccs()
# [["A", "C", "B"], ["D"]]  # SCCs with size > 1 are cycles

# Get cycle info
cycles = tarjan.find_cycles()
for cycle in cycles:
    print(f"Cycle: {cycle.nodes}")
    print(f"Length: {cycle.length}")
    print(f"Self-loop: {cycle.is_self_loop}")
```

### TopologicalSort

실무 운영 가이드에서 Topological, Kahn을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import TopologicalSort

dependencies = {
    "task_a": ["task_b", "task_c"],  # a before b, c
    "task_b": ["task_d"],
    "task_c": ["task_d"],
    "task_d": [],
}

sorter = TopologicalSort(dependencies)

# Perform sort
try:
    order = sorter.sort()  # ["task_a", "task_b", "task_c", "task_d"]
except ValueError:
    print("Cycle exists")

# Check for cycles
if sorter.has_cycles():
    print("Cycle detected")
```

### GraphTraversalMixin

실무 운영 가이드에서 Mixin을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import GraphTraversalMixin

class HierarchyValidator(BaseValidator, GraphTraversalMixin):
    def validate(self, df):
        # Build adjacency list
        adj = self.build_adjacency_list(
            df,
            id_column="id",
            parent_column="parent_id",
            cache_key="my_hierarchy",  # Caching key
        )

        # Find all cycles
        cycles = self.find_all_cycles(adj)

        # Find hierarchy cycles (parent-child relationships)
        child_to_parent = self.build_child_to_parent(df, "id", "parent_id")
        hierarchy_cycles = self.find_hierarchy_cycles(
            child_to_parent,
            max_depth=1000,
        )

        # Compute node depths
        depths = self.compute_node_depths(adj, roots=["root"])

        # Topological sort
        sorted_nodes = self.topological_sort(adj)

        # Clear cache
        self.clear_adjacency_cache()
```

## Covariance Computation Optimization

실무 운영 가이드에서 Efficient, Mahalanobis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### IncrementalCovariance

실무 운영 가이드에서 Incremental, Welford을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import IncrementalCovariance

# Incremental covariance for 10 features
cov = IncrementalCovariance(n_features=10)

# Update with streaming data
for batch in data_stream:
    cov.update_batch(batch)  # Batch update

# Single sample update
cov.update(sample)

# Get results
result = cov.get_result(regularization=1e-6)
print(result.mean)       # Mean vector
print(result.covariance) # Covariance matrix
print(result.n_samples)  # Sample count
```

### WoodburyCovariance

실무 운영 가이드에서 Woodbury을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import WoodburyCovariance

# Create from data
cov = WoodburyCovariance.from_data(
    training_data,
    regularization=1e-6,
)

# Efficient sample add/remove
cov.add_sample(new_sample)
cov.remove_sample(old_sample)

# Compute Mahalanobis distance
distance = cov.mahalanobis(query_point)
distances = cov.mahalanobis_batch(query_points)
```

### RobustCovarianceEstimator

실무 운영 가이드에서 MCD, Minimum, Covariance, Determinant을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import RobustCovarianceEstimator

estimator = RobustCovarianceEstimator(
    contamination=0.1,    # Expected outlier ratio
    n_subsamples=10,      # Number of subsamples
    subsample_size=500,   # Each subsample size
    random_state=42,
)

result = estimator.fit(large_data)
# result.is_robust = True
```

### BatchCovarianceMixin

실무 운영 가이드에서 Mixin을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import BatchCovarianceMixin

class MahalanobisValidator(BaseValidator, BatchCovarianceMixin):
    def validate(self, data):
        # Automatically select optimal method
        result = self.compute_covariance_auto(
            data,
            use_robust=True,
        )

        # Compute Mahalanobis distances
        distances = self.compute_mahalanobis_distances(data, result)

        # Create Woodbury covariance (for efficient updates)
        woodbury = self.create_woodbury_covariance(data)
```

**설정 Attributes:**

| 실무 운영 가이드에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `_batch_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Batch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `_robust_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `_regularization`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Regularization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Geographic Operations Optimization

실무 운영 가이드에서 Vectorized을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

        # Vectorized Haversine distance
        distances = self.haversine_vectorized(
            lat1=lats,
            lon1=lons,
            lat2=target_lat,
            lon2=target_lon,
            unit=DistanceUnit.KILOMETERS,
        )

        # Pairwise distance matrix
        dist_matrix = self.pairwise_distances(
            lats1, lons1, lats2, lons2,
            unit=DistanceUnit.METERS,
        )

        # Memory-efficient chunked computation
        for start, end, chunk in self.pairwise_distances_chunked(
            lats1, lons1, lats2, lons2,
            chunk_size=50000,
        ):
            process_chunk(chunk)

        # Find nearest point
        idx, dist = self.nearest_point(
            query_lat, query_lon, lats, lons,
        )

        # k-nearest neighbors
        indices, distances = self.k_nearest_points(
            query_lat, query_lon, lats, lons, k=5,
        )

        # Find points within radius
        indices, distances = self.points_within_radius(
            center_lat, center_lon, lats, lons,
            radius=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # Bearing calculation
        bearings = self.bearing_vectorized(lat1, lon1, lat2, lon2)

        # Destination point calculation
        dest_lat, dest_lon = self.destination_point(
            lat, lon,
            bearing=45,
            distance=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # Create bounding box
        bbox = self.create_bounding_box(
            center_lat, center_lon,
            radius=50,
            unit=DistanceUnit.KILOMETERS,
        )

        # Filter by bounding box
        mask, filtered_lats, filtered_lons = self.filter_by_bounding_box(
            lats, lons, bbox,
        )

        # Coordinate validation
        valid_mask = self.validate_coordinates(lats, lons)
```

### DistanceUnit

실무 운영 가이드에서 Supported을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Unit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Earth, Radius을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|--------------|
| 실무 운영 가이드에서 `METERS`, METERS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `KILOMETERS`, KILOMETERS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MILES`, MILES을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NAUTICAL_MILES`, NAUTICAL_MILES을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SpatialIndexMixin

실무 운영 가이드에서 Efficient, BallTree을(를) 다루는 항목입니다:

```python
from truthound.validators.optimization import SpatialIndexMixin

class IndexedGeoValidator(BaseValidator, SpatialIndexMixin):
    def setup(self, reference_coords):
        # Build spatial index
        self.build_spatial_index(
            lats=reference_coords[:, 0],
            lons=reference_coords[:, 1],
            leaf_size=40,
        )

    def validate(self, query_coords):
        # Query within radius
        results = self.query_radius(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            radius_km=10,
        )

        # k-nearest neighbor query
        distances_km, indices = self.query_nearest(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            k=5,
        )

        # Clear index
        self.clear_spatial_index()
```

## Aggregation Optimization

실무 운영 가이드에서 Polars, Leverage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### LazyAggregationMixin

```python
from truthound.validators.optimization import LazyAggregationMixin
import polars as pl

class CrossTableValidator(BaseValidator, LazyAggregationMixin):
    def validate(self, orders, order_items):
        # Lazy aggregation
        result = self.aggregate_lazy(
            lf=order_items.lazy(),
            group_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("total_qty"),
                pl.col("price").sum().alias("total_price"),
            ],
            cache_key="order_totals",  # Caching
        )

        # Join and aggregate in one operation
        result = self.aggregate_with_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            left_on="order_id",
            right_on="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            how="left",
        )

        # Streaming join for large tables
        result = self.streaming_aggregate_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            join_key="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            slice_size=100000,
        )

        # Compare aggregate values
        mismatches = self.compare_aggregates(
            source=orders,
            aggregated=result,
            key_column="order_id",
            source_column="expected_total",
            agg_column="total_qty",
            tolerance=0.01,
        )

        # Incremental aggregation
        updated = self.incremental_aggregate(
            existing=previous_result,
            new_data=new_items.lazy(),
            group_by="order_id",
            sum_columns=["quantity", "price"],
            count_column="item_count",
        )

        # Window aggregation
        with_windows = self.window_aggregate(
            lf=order_items.lazy(),
            partition_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("order_total"),
                pl.col("price").mean().alias("order_avg_price"),
            ],
        )

        # Semi-join filter
        filtered = self.semi_join_filter(
            main=orders.lazy(),
            filter_by=active_orders.lazy(),
            on="order_id",
            anti=False,  # True for anti-join
        )

        # Multi-table aggregation
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

        # Clear cache
        self.clear_aggregation_cache()
```

### AggregationExpressionBuilder

실무 운영 가이드에서 Fluent을(를) 다루는 항목입니다:

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

# Usage
result = lf.group_by("category").agg(exprs)
```

## 검증기 프로파일링

실무 운영 가이드에서 Measure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 프로파일러Config

프로파일러 설정:

```python
from truthound.validators.optimization import (
    ProfilerConfig,
    ProfilerMode,
)

# Basic configuration (timing only)
config = ProfilerConfig.basic()

# Standard configuration (timing + memory)
config = ProfilerConfig()  # Default

# Detailed configuration (with snapshots)
config = ProfilerConfig.detailed()

# Diagnostic configuration (maximum detail)
config = ProfilerConfig.diagnostic()

# Custom configuration
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

### Validator프로파일러

```python
from truthound.validators.optimization import (
    ValidatorProfiler,
    ProfilerConfig,
)

# Create profiler
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())

# Start session
profiler.start_session("validation_run_1", attributes={"env": "prod"})

# Profile validators
for validator in validators:
    with profiler.profile(validator, rows_processed=100000) as ctx:
        issues = validator.validate(lf)
        ctx.set_issue_count(len(issues))
        ctx.add_attribute("columns", ["a", "b", "c"])

# End session
session = profiler.end_session()

# Analyze results
print(session.summary())
print(session.to_json())

# Slowest validators
slowest = profiler.get_slowest_validators(n=10)
# [("SlowValidator", 150.5), ("AnotherValidator", 100.2), ...]

# Memory-intensive validators
memory_heavy = profiler.get_memory_intensive_validators(n=10)

# Overall summary
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

### Convenience Functions

```python
from truthound.validators.optimization import (
    profile_validator,
    profiled,
    get_default_profiler,
)

# Context manager
with profile_validator(my_validator, rows_processed=10000) as ctx:
    issues = my_validator.validate(lf)
    ctx.set_issue_count(len(issues))

print(ctx.metrics.timing.mean_ms)

# Decorator
class MyValidator(Validator):
    @profiled(track_issues=True)
    def validate(self, lf):
        return [issue1, issue2]

# Global profiler
profiler = get_default_profiler()
```

### Validator메트릭

실무 운영 가이드에서 Collected을(를) 다루는 항목입니다:

```python
metrics = profiler.get_metrics("MyValidator")

# Timing metrics
print(metrics.timing.mean_ms)    # Mean execution time
print(metrics.timing.median_ms)  # Median
print(metrics.timing.p95_ms)     # 95th percentile
print(metrics.timing.p99_ms)     # 99th percentile
print(metrics.timing.std_ms)     # Standard deviation

# Memory metrics
print(metrics.memory.mean_peak_mb)       # Mean peak memory
print(metrics.memory.max_peak_mb)        # Maximum peak memory
print(metrics.memory.total_gc_collections)  # GC collection count

# Throughput metrics
print(metrics.throughput.mean_rows_per_sec)  # Rows per second
print(metrics.throughput.total_rows)         # Total rows processed

# Issue metrics
print(metrics.total_issues)  # Total issues found
print(metrics.mean_issues)   # Mean issues
print(metrics.error_counts)  # Error counts
```

### 프로파일링Report

리포트 generation:

```python
from truthound.validators.optimization import ProfilingReport

report = ProfilingReport(profiler)

# Text summary
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

# HTML report
html = report.html_report()
with open("profile_report.html", "w") as f:
    f.write(html)

# Prometheus format export
prometheus_metrics = profiler.to_prometheus()
# # HELP validator_execution_duration_ms Validator execution duration
# # TYPE validator_execution_duration_ms gauge
# validator_execution_duration_ms_mean{validator="MyValidator",category="completeness"} 150.500
# ...
```

## 통합 Example

실무 운영 가이드에서 Example을(를) 다루는 항목입니다:

```python
import polars as pl
from truthound.validators.optimization import (
    # DAG orchestration
    ValidatorDAG,
    ParallelExecutionStrategy,
    # Profiling
    ValidatorProfiler,
    ProfilerConfig,
    # Mixins
    GraphTraversalMixin,
    BatchCovarianceMixin,
    VectorizedGeoMixin,
    LazyAggregationMixin,
)
from truthound.validators import NullValidator, RangeValidator


# Define custom validator
class OptimizedHierarchyValidator(
    BaseValidator,
    GraphTraversalMixin,
    BatchCovarianceMixin,
):
    def validate(self, df):
        # Graph cycle detection
        adj = self.build_adjacency_list(df, "id", "parent_id")
        cycles = self.find_all_cycles(adj)

        # Covariance for outlier detection
        numeric_data = df.select(pl.col(pl.Float64)).to_numpy()
        cov_result = self.compute_covariance_auto(numeric_data, use_robust=True)
        distances = self.compute_mahalanobis_distances(numeric_data, cov_result)

        return []


# Configure profiler
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())
profiler.start_session("optimized_validation")

# Build DAG
dag = ValidatorDAG()
dag.add_validators([
    NullValidator(),
    RangeValidator(min_value=0),
    OptimizedHierarchyValidator(),
])

# Create and execute plan
plan = dag.build_execution_plan()
print(plan.get_summary())

# Parallel execution
strategy = ParallelExecutionStrategy(max_workers=4)

df = pl.DataFrame({"id": [1, 2, 3], "parent_id": [None, 1, 2], "value": [10, 20, 30]})

with profiler.profile(plan, rows_processed=len(df)) as ctx:
    result = plan.execute(df.lazy(), strategy)
    ctx.set_issue_count(len(result.all_issues))

# End session and report
session = profiler.end_session()
print(session.to_json())
```

## API 레퍼런스

### orchestrator Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `ValidatorDAG`, ValidatorDAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증기 dependency DAG |
| 실무 운영 가이드에서 `ValidatorNode`, ValidatorNode을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증기 node wrapper |
| 실무 운영 가이드에서 `ValidatorPhase`, ValidatorPhase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ExecutionPlan`, ExecutionPlan을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ExecutionLevel`, ExecutionLevel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Group을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ExecutionResult`, ExecutionResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Execution 결과 |
| 실무 운영 가이드에서 `ExecutionStrategy`, ExecutionStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SequentialExecutionStrategy`, SequentialExecutionStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sequential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ParallelExecutionStrategy`, ParallelExecutionStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AdaptiveExecutionStrategy`, AdaptiveExecutionStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Adaptive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `create_execution_plan()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Convenience을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `execute_validators()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Convenience을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### graph Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `IterativeDFS`, IterativeDFS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Iterative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TarjanSCC`, TarjanSCC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tarjan, SCC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TopologicalSort`, TopologicalSort을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Topological을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `GraphTraversalMixin`, GraphTraversalMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Graph을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CycleInfo`, CycleInfo을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cycle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `NodeState`, NodeState을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Node을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### covariance Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `IncrementalCovariance`, IncrementalCovariance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Welford을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WoodburyCovariance`, WoodburyCovariance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Woodbury을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `RobustCovarianceEstimator`, RobustCovarianceEstimator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MCD-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BatchCovarianceMixin`, BatchCovarianceMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Batch을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CovarianceResult`, CovarianceResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Covariance 결과 dataclass |

### geo Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `VectorizedGeoMixin`, VectorizedGeoMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Vectorized을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SpatialIndexMixin`, SpatialIndexMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Spatial을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `DistanceUnit`, DistanceUnit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Distance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `BoundingBox`, BoundingBox을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bounding을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### aggregation Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `LazyAggregationMixin`, LazyAggregationMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Lazy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AggregationExpressionBuilder`, AggregationExpressionBuilder을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Aggregation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `AggregationResult`, AggregationResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Aggregation 결과 dataclass |
| 실무 운영 가이드에서 `JoinStrategy`, JoinStrategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Join strategy 설정 |

### 프로파일링 Module

| 실무 운영 가이드에서 Class/Function을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------------|-------------|
| 실무 운영 가이드에서 `ValidatorProfiler`, ValidatorProfiler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Main을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ProfilerConfig`, ProfilerConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일러 설정 |
| 실무 운영 가이드에서 `ProfilerMode`, ProfilerMode을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일러 mode enum |
| 실무 운영 가이드에서 `ValidatorMetrics`, ValidatorMetrics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증기 metrics |
| 실무 운영 가이드에서 `TimingMetrics`, TimingMetrics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Timing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MemoryMetrics`, MemoryMetrics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ThroughputMetrics`, ThroughputMetrics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Throughput을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ProfilingSession`, ProfilingSession을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 session |
| 실무 운영 가이드에서 `ExecutionSnapshot`, ExecutionSnapshot을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ProfilingReport`, ProfilingReport을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 리포트 generator |
| 실무 운영 가이드에서 `profile_validator()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 context manager |
| 실무 운영 가이드에서 `profiled()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 decorator |
