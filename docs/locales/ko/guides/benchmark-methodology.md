# 벤치마크 측정 방법

## Purpose

Truthound 벤치마크는 공개된 제로 설정 경로가 정확성을 유지하면서 비교 가능한 릴리스 등급 워크로드에서 Great Expectations보다 빠르게 실행되는지 검증합니다.

## Principles

- 내부 도우미가 아닌 공개 API를 측정합니다.
- 저장소에서 관리하는 워크로드 manifest와 fixture를 사용합니다.
- 첫 실행의 베이스라인 생성 비용과 반복 실행의 정상 상태 비용을 구분합니다.
- 각 프레임워크를 별도 자식 프로세스에서 실행합니다.
- 정확성이 유지되지 않는 성능 개선은 인정하지 않습니다.

## Measurement Model

프레임워크와 워크로드의 각 observation에는 다음을 기록합니다.

- 프레임워크 이름과 버전
- 워크로드 ID와 데이터셋 fingerprint
- backend와 exactness 등급
- cold 실행 시간
- warm 실행 시간의 중앙값
- 모든 warm 실행의 wall-clock 시간과 process CPU 시간
- cold 실행과 모든 warm 실행 각각의 정확성
- 최대 RSS 메모리 사용량
- 예상 issue 수와 실제 issue 수
- 워크로드별 실행 artifact 경로
- methodology, 설정한 worker thread budget, 실제 Polars thread pool 크기
- 데이터셋 fingerprint와 별도로 기록한 워크로드 계약 fingerprint

Truthound는 cold와 warm 실행에 같은 제로 설정 workspace를 사용합니다. 베이스라인 생성 비용은 cold에, 베이스라인 재사용은 warm 중앙값에 반영됩니다.

## Versioned Measurement Contract

새 실행의 기본값은 `BenchmarkMethodology.name = "truthound-parity-thread-budget-v2"`, cold 1회, `warm_iterations = 7`, `worker_threads = 1`입니다. Python runner에는 명시적 methodology를 전달할 수 있지만, 값을 바꾸면 비교 계약도 달라집니다. 사용자 지정 methodology는 참고용이며 공식 릴리스 판정이 아닙니다. `release-ga:measurement-policy`는 threshold를 포함한 전체 기본 `BenchmarkMethodology()`와의 일치를 요구합니다. 공개 wheel 검증기도 엄격한 기본값을 사용합니다. `truthound benchmark parity`의 새 CLI 옵션은 필요하지 않습니다.

Observation metadata는 `warm_durations_seconds`, `warm_process_cpu_seconds`, `iteration_correctness`, `worker_threads`, `polars_thread_pool_size`, `workload_contract_sha256`를 보존합니다. Warm 중앙값에는 가장 빠른 값이나 선택한 일부가 아닌 모든 warm 측정값을 사용합니다. Cold와 모든 warm 실행이 정확해야 하며, 마지막 실행이 정확하다고 앞선 오류가 사라지지는 않습니다. 데이터에 실제 품질 실패가 예상되는 경우 그 실패를 그대로 유지합니다. 정확성은 워크로드의 예상 결과와 일치한다는 뜻이지 모든 validation을 통과로 바꾼다는 뜻이 아닙니다.

누락·비유한 수치·불일치·불완전한 측정 기록은 검증에 실패합니다. 호스트 부하 값은 측정 환경을 설명할 뿐, 호스트가 유휴 상태였거나 시간이 안정적이었다는 증거가 아닙니다.

과거 `truthound-3.0-parity-gate` artifact는 원래 의미를 보존합니다. Warm 2회와 당시 필드가 없어 지정되지 않았던 thread budget을 v2로 바꾸거나 덮어쓰지 않습니다. 베이스라인 비교는 methodology, 워크로드 계약 fingerprint, 데이터셋 fingerprint, backend, exactness가 모두 일치해야 합니다. 호환되지 않는 베이스라인은 비교 불가로 거부하며 속도 향상이나 회귀 수치로 보고하지 않습니다.

## Child Process Isolation

실행 시간과 메모리는 부모 runner가 아닌 자식 프로세스에서 수집합니다. 이를 통해 다음 영향을 줄입니다.

- 이미 import한 모듈
- 앞선 워크로드에서 남은 allocator 상태
- 서로 다른 프레임워크의 캐시 혼합
- 공유 프로세스의 RSS 증가

어느 프레임워크든 자식 프로세스가 라이브러리를 import하기 전에 같은 thread budget을 적용합니다. 대상 환경 변수는 `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `TRUTHOUND_BENCHMARK_WORKER_THREADS`입니다. 실제 Polars pool 크기는 설정한 budget과 일치해야 합니다. 이는 참여 라이브러리의 pool 크기를 제한하는 것이며, 운영체제 thread가 하나뿐이거나 다른 호스트 작업의 영향이 없다는 뜻은 아닙니다.

## Runner Policy

Truthound는 두 종류의 runner를 사용합니다.

- GitHub-hosted nightly runner: 추세와 조기 이상 징후 확인
- 고정 self-hosted runner: 공식 릴리스 등급 벤치마크 검증

Nightly artifact는 참고용이며, 고정 runner의 release artifact가 공식 판정 근거입니다. 공식 판정에는 CPU 모델, 논리 코어 수, RAM, OS, Python minor 버전, 저장장치 종류를 기록해야 합니다.

릴리스 workflow는 고정 호스트 정보와 다음 환경 계약을 사용합니다.

- `TRUTHOUND_BENCHMARK_RUNNER_CLASS=self-hosted-fixed`
- `TRUTHOUND_BENCHMARK_RUNNER_LABELS=self-hosted,benchmark-fixed`
- `TRUTHOUND_BENCHMARK_RELEASE_VERDICT=true`
- `TRUTHOUND_BENCHMARK_STORAGE_CLASS`
- `TRUTHOUND_BENCHMARK_CPU_MODEL`
- `TRUTHOUND_BENCHMARK_RAM_BYTES`
- `TRUTHOUND_BENCHMARK_CPU_PHYSICAL_CORES`

Self-hosted macOS runner에서는 `actions/setup-python` 대신 `uv`로 관리하는 Python 3.11과 해당 실행 전용 가상환경을 사용합니다. 각 workflow run과 attempt가 별도 환경을 사용하며 공유 `.release-venv`를 초기화하지 않습니다. 이를 통해 macOS에서 `/Users/runner` 같은 hosted-toolcache 경로를 가정하는 문제도 피합니다.

## Thresholds

릴리스 등급 기준은 다음과 같습니다.

- local exact: `Truthound >= 1.5x Great Expectations`
- SQL exact: `Truthound >= 1.0x Great Expectations`, 목표는 `1.2x`
- local memory: `Truthound <= 60%` of Great Expectations peak RSS

호환되는 기존 Truthound baseline과의 비교에서는 warm 중앙값의 최대 회귀 허용치 `10%`를 그대로 유지합니다.

정확성 parity가 실패하면 시간이 더 빨라도 성능 비교는 실패입니다. V2 측정 방식 정정은 이 기준을 낮추지 않습니다. 기존 공개 수치는 과거 결과로 유지하며, 새로운 성능 주장은 명시한 methodology로 실제 검증한 새 artifact set이 있어야 합니다.

## Published Wheel Verification

릴리스 workflow는 source checkout 검증과 이미 공개된 wheel 검증을 구분합니다. `scripts/benchmarks/verify_release_package.py`는 후자를 준비·실행하며, 수정된 벤치마크 도구를 사용하더라도 설치된 Truthound 제품을 checkout 코드로 대체하지 않습니다.

공개 wheel lane은 릴리스 workflow를 `target=released-wheel`로 dispatch하고 `release_version`, `release_tag`, `release_commit`(소문자 16진수 40자), `wheel_sha256`(소문자 16진수 64자)을 명시합니다. Dispatch한 workflow revision이 harness commit입니다. 릴리스 tag와 PyPI wheel은 변경하지 않으며 기존 source checkout lane은 별도로 유지합니다.

공개 wheel의 provenance에는 릴리스 버전·tag·commit, wheel SHA-256, 실제 설치 패키지 정보, harness commit, methodology, 워크로드 manifest digest를 따로 기록합니다. Harness는 별도 namespace에 로드하고, 측정할 공개 API는 검증된 설치 wheel에서 가져옵니다. Worker가 checkout이나 user site package를 대신 import해서는 안 됩니다.

JSON 결과의 `metadata.release_package`에는 `harness_files`, `harness_sha256`, 내용 기반 주소로 식별한 `workloads` manifest를 포함한 근거를 기록합니다. `measurement_target="immutable-installed-wheel"`, `harness_namespace="_truthound_release_harness"`로 측정 대상을 구분합니다. 각 observation의 `metadata.release_package_provenance`는 이 provenance 객체의 digest를 연결하고 `metadata.release_package_measurement_integrity`로 측정 무결성을 기록합니다. Workflow는 설치 의존성의 이름과 버전만 `installed-dependencies.json`에 기록하며, freeze 출력에 포함될 수 있는 패키지 원본 URL은 보존하지 않습니다.

변경하지 않은 `3.1.15` wheel을 수정된 harness로 검증한 결과는 그 wheel과 해당 harness 조합의 근거입니다. 수정된 source checkout이 `3.1.15`로 공개됐다는 뜻이 아닙니다. 기존 릴리스 artifact를 대체하거나 이전 실패를 성공으로 바꿔 기록하지 않습니다. Provenance 확인과 실제 정확성·시간·메모리 gate가 모두 충족돼야 릴리스 검증 판정을 내릴 수 있습니다.

## Commands

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local
truthound benchmark parity --suite nightly-core --frameworks both --backend local
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite
truthound benchmark parity --suite release-ga --frameworks both --strict
```

`release-ga`는 nightly suite보다 엄격합니다.

- `--frameworks both`를 사용해야 합니다.
- `--backend` 필터를 사용하면 안 됩니다.
- 8개의 tier-1 local·SQLite 워크로드를 모두 실행해야 합니다.
- 변경하지 않은 threshold를 포함한 전체 기본 v2 methodology를 사용해야 합니다.

## Artifact Layout

벤치마크 artifact root는 활성 프로젝트의 `.truthound/benchmarks/`입니다.

- `results/`
- `baselines/`
- `artifacts/`
- `release/`

Parity 실행에서 외부 출력 파일을 지정하면 그 옆에 다음 canonical artifact set을 생성합니다.

- `release-ga.json`
- `release-ga.md`
- `release-ga.html`
- `env-manifest.json`
- `latest-benchmark-summary.md`

승인된 릴리스 artifact set으로 문서 요약 페이지를 발행하는 명령은 다음과 같습니다.

```bash
python docs/scripts/publish_benchmark_summary.py \
  --json benchmark-artifacts/release-ga.json \
  --artifact-base-url .. \
  --output docs/releases/latest-benchmark-summary.md
```

## Related Reading

- [성능과 벤치마크](performance.md)
- [Great Expectations 비교](gx-parity.md)
- [워크로드 목록](benchmark-workloads.md)
- [문서 배포 검증](docs-deployment-verification.md)
