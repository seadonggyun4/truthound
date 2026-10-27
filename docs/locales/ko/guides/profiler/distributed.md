# 분산 프로파일링

`truthound.profiler.distributed` 패키지는 `DistributedProfiler`,
`DistributedProfileConfig`, `BackendType`, `PartitionStrategy`와 native
`LocalBackend`, `DaskBackend`, `RayBackend`, `SparkBackend`를 제공합니다.
`DistributedBackend`는 enum이 아닌 backend 인터페이스입니다.

```python
import polars as pl
from truthound.profiler.distributed import DistributedProfiler, DaskConfig

with DistributedProfiler.create(
    backend="dask",
    backend_config=DaskConfig(scheduler="threads"),
    num_partitions=2,
) as profiler:
    profile = profiler.profile(pl.DataFrame({"x": [1.0, None, 3.0]}), name="measurements")
    print(profile.row_count, profile.columns[0].null_count)
```

backend 설정은 `backend_config`, 상위 설정은 `DistributedProfileConfig` 또는
`create` 키워드로 전달합니다. backend 이름은 `local`, `dask`, `ray`, `spark`,
`auto`입니다. 자동 선택은 설치된 Dask, Ray, Spark, local 순이며 자원 소유권을
예측하려면 명시적으로 선택하세요. 해당 `dask`, `ray`, `spark` extra와 native
런타임 선행 조건을 설치해야 합니다. import 성공은 클러스터 준비 확인이 아닙니다.

## 입력과 전역 통계

입력은 Polars DataFrame/LazyFrame 또는 backend가 지원하는 파일 경로입니다.
메모리 Polars 입력은 배포 전에 수집하므로 임의 크기 입력에 대해 driver 메모리
상한을 보장하지 않습니다. Local·Dask는 CSV/Parquet를 지원하며 Ray·Spark 경로
입력은 Parquet입니다. 파티션 및 전역 집계 동안 원본을 안정적으로 유지하세요.

Local 열 분할은 테이블 행을 한 번만 계산합니다. 행 분할 수가 입력 행 수보다
커도 행 수를 부풀리지 않습니다. 전역 고유값 수는 null 범주를 포함합니다.
숫자 평균·표본 표준편차는 null을 제외하며 문자열 길이 최소·최대·평균은 전체
non-null 행을 사용합니다. Local은 native Polars 전역 집계, Dask는 native
전역 표현식, Ray는 소유 remote task의 전역 Polars 집계, Spark는 native SQL
집계를 사용합니다. 고유값·문자열 길이를 첫 파티션에서 가져오지 않습니다.
추가 읽기와 backend shuffle 상태가 필요할 수 있습니다.

실패·누락·ID 불일치 파티션은 부분 성공 `TableProfile` 대신 오류를 반환합니다.
`timeout_seconds`는 native 작업 반환 후 늦은 결과를 거부하며 쿼리 강제 중단이
아닙니다. 명시적 클러스터 취소와 전송 실패는 native backend 동작을 따릅니다.

## 런타임 소유권

context manager로 소유 자원을 초기화·종료하세요. Dask는 계산마다 scheduler를
선택하며 전역 scheduler를 변경하지 않습니다. 종료 시 client와 소유 cluster를
모두 정리 시도하고 실패를 함께 전달합니다. 본문 실패와 종료 실패도 함께
보존합니다. Ray는 이 backend가 시작한 런타임만 종료합니다. Spark도 이
backend가 만든 context만 중단하여 이미 활성화된 context를 보존합니다.
독립 JVM·Ray 수명에는 driver subprocess 격리를 권장합니다.

## 표본·패턴·callback·MCP

`sample_size`는 프로파일링 전 native 입력의 앞부분으로 제한하며 반환 행 수는
그 표본을 기술합니다. Spark 분산 행 순서는 원본이 순서를 제공하지 않으면
안정적이지 않습니다. `include_patterns=True`는 각 문자열 열에서 기본 최대
1,000개 non-null 값을 Core `PatternAnalyzer`로 분석하며 false는 패턴 표본을
읽지 않습니다. native 패턴 결과에는 표본 값이 포함되므로 민감한 프로파일
내용으로 취급해야 합니다.

`map_partitions(callback, partitions, data)`는 operator가 제공한 직렬화 가능한
callback에 `PartitionInfo`와 이미 선택된 Polars DataFrame을 전달합니다.
같은 파티션 ID와 실제 입력 행 수를 가진 `WorkerResult`를 반환해야 합니다.
native 프로파일링도 같은 callback 경로를 사용합니다. 예외는 실패로 유지되며
native 전역 통계 reducer는 사용자 metadata의 범용 reducer가 아닙니다.
한 profiler의 호출은 backend 활성 입력 접근을 직렬화합니다.

승인된 factory는 초기화한 `DistributedProfiler`를
`ProfilerRuntime.profilers`에 등록할 수 있습니다. 기존 MCP `profile`, 프로파일
저장·복구, 보관 표본 프로파일링·해제 경로가 native 객체를 사용합니다. factory가
backend 초기화·정리를 소유하며 wire 요청은 alias와 고정 asset만 선택합니다.
클러스터 주소·자격증명·callback 코드는 전달하지 않습니다. native 실행 guard가
프로파일링 단계 사이 원본 권한을 확인합니다.

## 분할·자원 정책

`ROW_BASED`는 행, `COLUMN_BASED`는 겹치지 않는 열 그룹, `HYBRID`는 행·열 그룹,
`HASH`는 전체 값이 같은 행을 같은 파티션에 배치합니다. hybrid 파티션 수는
계획 힌트이며 요청 수와 달라질 수 있습니다. native hash와 행 순서는 엔진·
버전에 따라 다르므로 엔진 간 동일한 bytes의 표본을 보장하지 않습니다.
Local·Ray는 공유 Polars 계획을, Dask·Spark는 native shuffle·열 계획을 사용합니다.
모든 정책은 프로파일에서 원본 기록이나 셀을 추가·누락하지 않습니다.

명시적 `num_partitions`가 없으면 `BackendConfig.parallelism`을 기본 분할 수로
사용합니다. `memory_per_worker`는 `512m`, `2g` 같은 양수 byte 크기를 받으며
callback 호출 전에 실제 수집 입력을 검사합니다. heap 예약은 아니므로 입력
로딩·정렬·shuffle·전역 집계는 더 많은 메모리를 사용할 수 있습니다. backend의
`timeout_seconds`는 callback 결과, 상위 timeout은 전체 프로파일 결과를
협력적으로 제한합니다. 0은 해당 제한을 해제합니다. 실행 중 native 작업은
강제 중단하지 않고 종료를 기다립니다.

Core는 `RetryableExecutionError` 이후에만 최대 `retry_count`번 재시도합니다.
일반 실패·늦은 결과는 Core 재실행을 요청하지 않습니다. Ray callback task의
암묵적 재시도는 끄며 Spark 클러스터 재시도는 운영자의 native Spark 설정을
따릅니다. 결과가 불명확한 외부 부수 효과를 profiler callback에 넣지 마세요.

## 영구 native checkpoint

선택 backend config에 `checkpoint_enabled=True`와 로컬 `checkpoint_dir`를
지정합니다. native Parquet 직렬화는 실행 엔진과 공유하는 checkpoint 게시기를
사용하며 SHA-256 inventory·파일/디렉터리 동기화·원자적 디렉터리 게시를
적용합니다. `backend.checkpoint_path`는 게시된 snapshot을 가리킵니다. 실패한
staging 디렉터리는 운영자가 확인할 수 있도록 남깁니다.

`backend.restore_checkpoint(location, num_partitions=..., strategy=...)`는 backend
식별자와 전체 inventory를 검증한 뒤 native Parquet를 읽고 `map_partitions`·
`aggregate_results`용 파티션 handle을 반환합니다. 새 backend가 다시 열 수
있으며 손상을 거부합니다. backend checkpoint 설정이 여전히 켜져 있으면
배포 시 새 snapshot도 게시합니다. checkpoint 경로에는 민감한 원본이 담기며
factory가 소유합니다. MCP profile 응답에 포함하지 않고 세션 해제 시 자동
삭제하지도 않습니다.


## 명시적 Arrow 변환 정책

고급 `truthound.execution.distributed.arrow_bridge.ArrowBridge`는 `ArrowBridgeConfig`를 복사해 보관합니다. 일반·백엔드별 변환에 같은 인덱스·null·시간·바이트 제한을 적용합니다. Pandas/Dask 인덱스는 `preserve_index`로 보존하며 다른 백엔드의 행 식별자를 생성하지는 않습니다. `batch_size`, 명시적 청크 크기, `max_memory_bytes`는 양의 정수여야 합니다. Dask는 파티션 하나를 변환한 뒤 행 수가 제한된 청크로 나누고 Spark는 선택한 전략으로 수집한 뒤 나눕니다. Ray는 네이티브 Arrow 배치를 소비합니다. 프로세스의 순간 최대 메모리 제한은 보장하지 않습니다. 바이트 제한은 네이티브 materialization과 정책 변환 뒤 Arrow `nbytes`를 검사하며 전체 결과 변환에서는 합친 크기도 제한합니다.

`null_handling="mask"`는 최상위 null을 보존하고 `"error"`는 거부합니다. `"sentinel"`은 null이 있는 각 열에 대해 명시적이고 null이 아니며 타입이 맞는 `sentinel_values`를 요구합니다. 시간 변환은 최상위 timestamp·duration·time을 마이크로초로, date64를 date32로 변환합니다. 안전한 cast를 사용하여 마이크로초 미만 정밀도를 버리지 않고 실패합니다. 원래 정밀도를 유지하려면 `coerce_temporal=False`를 선택하세요. 중첩된 시간·null 필드는 네이티브 표현을 유지합니다. 필드·스키마 메타데이터도 보존합니다. `collect=False`는 변환 검사가 끝난 뒤 LazyFrame 반환 타입을 선택하며 원본 materialization을 지연하지 않습니다.

Spark 변환은 호출자의 Arrow 세션 설정을 바꾸지 않으며 임의의 네이티브 실패를 다른 변환 경로로 재시도하지 않습니다. 명시적 native 전략에서 collector가 없으면 실패하고, 자동 선택에서는 해당 기능 부재 시 Pandas 경로를 사용할 수 있습니다. Polars에서 Spark로 변환할 때는 정책 검사 뒤 Pandas 기반 생성 호출을 한 번 수행합니다. 외부 클러스터와 원격 소스 서비스는 별도의 실제 검증이 필요합니다.
