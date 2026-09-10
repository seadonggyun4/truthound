# `truthound benchmark parity`

Truthound의 성능과 정확성을 검증하기 위해 저장소에서 관리하는 parity suite를 실행합니다.

## Usage

```bash
truthound benchmark parity [OPTIONS]
```

## Key Options

| 옵션 | 의미 |
| --- | --- |
| `--suite` | `pr-fast`, `nightly-core`, `nightly-sql`, `release-ga` 중 선택 |
| `--frameworks` | `truthound`, `gx`, `both` 중 선택 |
| `--backend` | 선택적 필터: `local`, `sqlite`, `duckdb-shadow` |
| `--output` | JSON artifact를 지정한 경로에 저장 |
| `--save-baseline` | suite 결과를 canonical parity baseline으로 저장 |
| `--compare-baseline` | 저장한 baseline과 Truthound 결과를 비교 |
| `--strict` | parity 누락, 기준 회귀, 요청한 프레임워크 사용 불가 시 실패 |

## Measurement and Baseline Compatibility

새 실행은 `truthound-parity-thread-budget-v2`를 사용합니다. Cold 1회와 warm 7회를 측정하며, 두 프레임워크 모두 자식 프로세스의 import 전에 worker thread budget 1을 적용합니다. 모든 warm wall-clock·CPU 측정값과 cold·warm 각 실행의 정확성을 보존합니다. 뒤의 실행이 정확하더라도 앞선 불일치를 숨기지 않습니다. 기존 정확성·속도·메모리 기준은 그대로 유지합니다.

이 기본값을 위한 새 CLI 옵션은 없습니다. `--compare-baseline`에는 methodology, 워크로드 계약 fingerprint, 데이터셋 fingerprint, backend, exactness가 일치해야 합니다. 호환되지 않으면 비교가 실패하며 속도 향상으로 보고하지 않습니다. 과거 artifact는 원래 methodology를 유지하고 v2로 덮어쓰지 않습니다.

## Examples

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
truthound benchmark parity --suite nightly-core --frameworks both --backend local --strict
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
truthound benchmark parity --suite release-ga --frameworks both --strict
```

## Release-Grade Verification Rules

`release-ga`는 고정 runner에서 공식 검증에 사용하는 suite이며, 다음 조건이 추가됩니다.

- `--frameworks`는 `both`여야 합니다.
- `--backend`를 지정하면 안 됩니다.
- Methodology는 warm 7회, 라이브러리 pool별 worker thread 1개, 변경하지 않은 threshold를 포함한 전체 기본 v2 계약과 정확히 같아야 합니다. 사용자 지정 Python API methodology는 참고용이며 공식 `release-ga:measurement-policy` assertion을 통과하지 못합니다.

## Artifacts

Parity 실행은 `.truthound/benchmarks/` 아래에 다음 artifact를 생성합니다.

- JSON 결과
- Markdown 요약
- HTML 요약
- `env-manifest.json`

JSON observation에는 설정한 thread budget, 실제 Polars pool 크기, 전체 warm 측정값, 각 반복의 정확성, 워크로드 계약 fingerprint가 포함됩니다. 검증기는 요약만 신뢰하지 않고 이 필드로 누락되거나 일치하지 않는 측정을 거부할 수 있습니다.

`release-ga`는 지정한 출력 경로 옆에 `latest-benchmark-summary.md`도 생성합니다.

변경하지 않은 공개 wheel을 수정된 harness로 검증하는 작업은 별도 릴리스 workflow lane이며 이 명령의 새 옵션이 아닙니다. Wheel 버전·digest와 harness revision을 구분해서 기록합니다. [공개 wheel 검증](../../guides/benchmark-methodology.md#published-wheel-verification)을 참고하세요.

## Related Reading

- [성능과 벤치마크](../../guides/performance.md)
- [벤치마크 측정 방법](../../guides/benchmark-methodology.md)
- [최신 검증 벤치마크 요약](../../releases/latest-benchmark-summary.md)
