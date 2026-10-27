# 샘플링

`truthound.profiler.sampling`은 Polars `LazyFrame`에서 행을 선택합니다. 행 수 계산과
표본 선택에는 동일하게 유지되는 소스를 사용하세요. 결과는 `data` LazyFrame, `metrics`,
`is_sampled`를 제공하며 collect하면 남은 계획을 실행합니다.

## SamplingConfig

```python
import polars as pl
from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod

frame = pl.DataFrame({"x": range(1000)}).lazy()
sampler = Sampler(SamplingConfig(
    strategy=SamplingMethod.RANDOM, max_rows=100, min_sample_size=100,
    seed=7, confidence_level=0.95, margin_of_error=0.05,
    max_memory_mb=16, enable_caching=True,
    fallback_strategy=SamplingMethod.HEAD,
))
result = sampler.sample(frame)
print(result.data.collect().height, result.metrics.sample_size)
```

`max_rows=0`은 행 상한을 없애지만 통계적 표본 크기 계산은 적용합니다.
`min_sample_size`도 모집단 크기와 양수 행 상한 이하로 제한됩니다. `max_memory_mb=0`은
출력 메모리 검사를 끕니다. 양수 값은 `Sampler.sample()` 반환 전에 구체화한 표본의 추정
바이트를 제한하며 native heap 예약이나 모든 소스 할당의 OOM 방지는 아닙니다.
`enable_caching`은 adaptive 전략의 제한된 선택 캐시이며 데이터·결과 캐시가 아닙니다.
각 sampler는 기본 설정을 복사합니다.

`SamplingConfig.for_accuracy("low" | "medium" | "high" | "maximum")`,
`for_speed()`, `for_memory(max_memory_mb=100)`를 사용할 수 있습니다.
`calculate_sample_size(population_size, confidence_level=0.95,
margin_of_error=0.05, min_sample_size=1)`은 유한 모집단 보정을 포함한 Cochran 공식을
사용합니다. 편향되거나 서로 독립적이지 않은 관측에서 정확도를 보증하지 않습니다.

## 전략

| 방식 | 동작 |
| --- | --- |
| `NONE` | 표본 행 상한을 무시하고 명시적으로 전체 행을 반환합니다. |
| `HEAD` | 계산한 수만큼 앞쪽 행을 선택하므로 순서 편향이 가능합니다. |
| `RANDOM` | seed를 사용한 행 인덱스 hash 순서로 계산한 크기만큼 선택합니다. |
| `SYSTEMATIC` | 별도 RNG의 시작 위치와 제한된 간격 선택을 사용합니다. |
| `HASH` | 설정 열 또는 기본 행 인덱스의 안정된 hash 순서로 선택하고 동률은 입력 순서를 사용합니다. |
| `STRATIFIED` | 그룹 비례 할당과 최대 나머지 반올림 후 그룹 안에서 seed로 선택합니다. |
| `RESERVOIR` | native chunk에 Algorithm R을 적용해 선택 행과 입력 chunk 하나를 유지합니다. |
| `ADAPTIVE` | 데이터 크기 기준으로 선택하며 작은 데이터도 양수 행 상한을 지킵니다. |

Random·hash는 native 정렬이 필요할 수 있습니다. 층화는 그룹별 크기를 구체화하여 quota를
join합니다. 예산이 부족하면 작은 그룹의 할당은 0일 수 있습니다. Reservoir 유지 공간은
O(표본 크기)이며 O(1)이 아닙니다. framework는 선택 전에 행 수를 계산하며 reservoir
선택 자체는 한 번 순회합니다. 현재 Polars는 `collect_batches`를 사용하지만 이 iterator가
없는 구버전은 결과를 구체화한 뒤 나누므로 동일한 스트리밍 메모리 제한을 제공하지 못합니다.
임시 열은 사용자 열을 대체하지 않으며 고정 seed는 전역 random 상태를 바꾸지 않습니다.

층화·hash 열은 `SamplingConfig`가 아닌 전략 객체에 지정합니다.

```python
from truthound.profiler.sampling import (
    SamplingStrategyRegistry, StratifiedSamplingStrategy,
)
registry = SamplingStrategyRegistry()
registry.register(StratifiedSamplingStrategy(stratify_column="category"))
sampler = Sampler(SamplingConfig(strategy=SamplingMethod.STRATIFIED, seed=7), registry)
```

`HashSamplingStrategy(hash_column="id")`와 신뢰된 사용자 `SamplingStrategy`도 등록할 수
있습니다. 일반 전략 오류에는 설정한 fallback을 사용하지만 `MemoryError`, `TimeoutError`,
`PermissionError`는 재조회 없이 전파합니다. fallback은 외부 callback 효과의 재실행 안전성을
보증하지 않습니다.

## SamplingMetrics

`original_size`, `sample_size`, `sampling_ratio`, `confidence_level`, `margin_of_error`,
`strategy_used`, `sampling_time_ms`, `memory_saved_estimate_mb`를 제공합니다.
`is_full_scan`, `reduction_factor`는 계산된 속성입니다. lazy 전략 시간은 계획 생성 시간을
측정하며 reservoir와 출력 검사는 실제 작업도 실행합니다. 메모리 절감은 추정이며 프로세스의
실제 최대 사용량이 아닙니다.

`sample_data(lf, max_rows=100_000, strategy="adaptive")`와 한 열을 먼저 선택하는
`Sampler.sample_column(lf, column, config=None)`도 제공합니다.

## 승인된 MCP 샘플링

비공개 `ProfilerRuntime.samplers`는 설정·registry를 포함한 native `Sampler`에 연결합니다.
`sampling_sample(sampler_id, asset_id)`는 고정 자산에서 한 번 표본을 만들고 구체화된
snapshot을 유지합니다. native 행 지표와 owner 메모리 예산을 확인하여 `sample_id`·지표를
반환합니다. `sample_get(sample_id, columns?, offset=0, limit=100)`은 페이지를,
`sample_profile(sample_id, profiler_id)`는 선택된 native profiler 결과를 제공합니다.
`sample_release(sample_id)`는 표본을 해제합니다. 표본은 최대 16개이며 owner 로컬로
worker 재시작을 넘기지 않습니다. 정확한 승인·민감 결과 동의가 필요하며 사용자 코드,
소스 경로, registry import를 wire로 받지 않습니다.

## 다음 단계

- [품질 점수](quality-scoring.md)
- [ML 추론](ml-inference.md)

## 병렬 블록 실행

`ParallelBlockSampler(ParallelSamplingConfig(...)).sample(frame, SamplingConfig(...))`는
소유 worker에서 선택 행을 실제 수집한 뒤 보관된 snapshot을 반환합니다.
블록 quota의 합은 요청 표본 크기와 같으며 마지막 짧은 블록도 비례 quota를
받습니다. seed `0`도 재현 가능하며 블록 순서와 원본 열을 보존합니다.
행 수 계산과 읽기 동안 입력을 안정적으로 유지해야 합니다.

`execution_mode=ExecutionMode.PROCESS`는 직렬화 가능한 Polars 계획과 spawn
프로세스를 사용합니다. 애플리케이션 진입점을 `if __name__ == "__main__":`로
보호하세요. `THREAD`와 `ADAPTIVE`는 스레드를 사용합니다. 제출 작업 수는
`max_workers * (1 + prefetch_blocks)`로 제한합니다.
`enable_work_stealing=True`이면 완료 작업의 빈자리를 바로 채우고 false이면
제출 window 전체 완료를 기다립니다. 공유 executor queue를 사용하므로
`blocks_stolen`은 추정값이 아닌 0입니다. priority 정책은 제출 전 블록
우선순위로 정렬하며 동일 우선순위는 블록 순서를 유지합니다. 프로세스 실행은
사용자 owner 메서드 override를 상속하지 않습니다.

블록 실패는 순차 재실행이나 부분 성공 없이 전파됩니다. 제한시간 이후 결과를
거부하고 진행 중 작업 종료를 기다리지만 실행 중 Polars 쿼리를 강제 중단하지는
않습니다. 전체 스캔을 포함한 보관 표본 크기는
`memory_budget.available_memory_mb` 이하여야 합니다. 할당 후 검사이므로
프로세스 heap 예약은 아닙니다. 모니터링은 RSS 초과 시 GC를 요청할 수 있으며
쿼리 정렬·프로세스 직렬화는 보관 표본보다 많은 메모리를 사용할 수 있습니다.
한 sampler의 호출은 직렬화됩니다.

승인된 profiler factory는 `truthound.mcp.profiler_services`의
`NativeSamplingBinding(strategy, SamplingConfig(...))`을
`ProfilerRuntime.samplers`에 넣을 수 있습니다. 선택 native strategy를 fallback
없이 호출합니다. `sampling_sample`은 실제 결과를 보관하여 `sample_get`,
`sample_profile`, `sample_release`에서 사용합니다. strategy·실행 모드·경로는
factory 설정으로 유지됩니다.

## Enterprise strategy와 표본 sketch

`BlockSamplingStrategy`는 worker 1개의 블록 executor를 재사용하며
`BlockSamplingMetrics`를 제공합니다. `MultiStageSamplingStrategy`는 각 축소
단계에서 정확한 순위 선택을 실행합니다(`num_stages >= 1`).
`ColumnAwareSamplingStrategy`는 열 복잡도로 크기를 조정하지만 양수인
`SamplingConfig.max_rows`를 넘지 않습니다. 세 경로 모두 반환 snapshot을
실제 수집하고 표본 메모리와 지연 결과를 검사합니다.

`ProgressiveSamplingStrategy`는 같은 seed로 점점 커지는 포함 관계 표본을
평가합니다. 단계 사이 각 열의 null 비율, 숫자 평균·모집단 표준편차 또는
비숫자 고유값 비율을 비교합니다. 최대
`abs(new-old) / max(1, abs(old))`가 `convergence_threshold` 이하이면 조기
종료합니다. `enable_progressive=False`는 중복 크기를 제외한 모든 단계를
완료합니다. `ProgressiveSamplingMetrics`는 실제 `stages_completed`,
`converged`, `convergence_delta`를 제공합니다. 전체 스캔은 일반
`SamplingMetrics`를 유지합니다. 이 수렴 휴리스틱은 통계적 정확도 보장이
아닙니다.

`SketchBasedSampler`는 값이 입력된 native
`cardinality_sketches[column]`(HyperLogLog)와
`frequency_sketches[column]`(CountMinSketch)을 가진 `SketchSamplingResult`를
반환합니다. 원본 모집단이 아닌 보관 표본의 sketch이며 각 native 추정·조회
API를 사용할 수 있습니다. MCP 표본 포트는 sample ID 아래 native sketch도 보관합니다.
`sample_sketch_query(sample_id, column, sketch_value)`는 null을 포함한 명시적
JSON scalar의 고유값·빈도 추정치를 반환하며 조회 값을 응답에 반복하지
않습니다. release는 행과 sketch를 함께 제거하고 owner 재시작 시 둘 다
만료됩니다. 보관 승인 예산은 행과 native sketch 메모리 추정치를 합산합니다.
sketch 생성 후 제한시간도 확인합니다. 보관 행 예산은 Python sketch table이나
임시 쿼리 할당을 포함하지 않습니다.
