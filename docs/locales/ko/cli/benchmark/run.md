# truthound 벤치마크 run

Run 성능 benchmarks.

## Synopsis

```bash
truthound benchmark run [BENCHMARK] [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `benchmark`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Specific을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--suite`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-s`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 벤치마크 suite (quick, ci, full, 프로파일링, 검증) |
| `--size` | | `small` | 행 수 프리셋: tiny, small, medium, large, xlarge, stress |
| CLI 명령 실행에서 `--rows`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-r`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--iterations`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-i`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--warmup`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-w`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Warmup을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--output`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Output 파일 path |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Auto을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-o`, Output, Auto-detected을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--save-baseline`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Save 결과 as baseline |
| CLI 명령 실행에서 `--compare-baseline`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | CLI 명령 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Compare을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `--verbose`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-v`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `false`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Verbose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

콘솔과 JSON 출력에는 선택적인 HTML 리포트 의존성이 필요하지 않습니다. HTML 출력을 사용하려면 `truthound[reports]`를 설치하세요.

## Description

CLI 명령 실행에서 `benchmark run`을(를) 다루는 항목입니다:

1. CLI 명령 실행에서 Prepares을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Warms을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. **Executes** 벤치마크 iterations
4. CLI 명령 실행에서 Collects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. **리포트** 결과 and statistics

## 벤치마크 Suites

| CLI 명령 실행에서 Suite을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Estimated, Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 벤치마크 Included |
|-------|---------------|-------------|---------------------|
| CLI 명령 실행에서 `quick`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Fast을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일, check, learn (1K rows) |
| CLI 명령 실행에서 `ci`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `full`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Core을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `profiling`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 프로파일링 focused | All 프로파일링 category benchmarks |
| CLI 명령 실행에서 `validation`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 검증 focused | All 검증 category benchmarks |

## Data Size Presets

| CLI 명령 실행에서 Size을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Rows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|------|--------|----------|
| CLI 명령 실행에서 `tiny`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `small`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Development을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `medium`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `large`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 성능 testing |
| `xlarge` | 약 10,000,000 | 약 5 GB | 대규모 성능 검증 |
| `stress` | 100,000,000 | 작업에 따라 달라짐 | 명시적 대규모 스트레스 검증 |

## 예시

### Run Specific 벤치마크

```bash
truthound benchmark run profile --size small
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
======================================================================
  BENCHMARK: single:profile
======================================================================

Environment: Python 3.13 on Darwin
Polars: 1.x, Truthound: 1.x

Results: 1/1 passed (100%)
Total Duration: 142.58ms

  [PROFILING]
    ✓ profile: 0.72ms (13.84M rows/s)

======================================================================
```

### Run 벤치마크 Suite

```bash
# Quick suite for fast verification (~5 seconds)
truthound benchmark run --suite quick

# CI/CD optimized suite (~15 seconds)
truthound benchmark run --suite ci

# Full benchmark suite (~30 seconds)
truthound benchmark run --suite full
```

### Custom Row Count

```bash
# Override size preset with exact row count
truthound benchmark run check --rows 1000000
```

### Custom Iterations

```bash
# More iterations for accuracy
truthound benchmark run profile --iterations 10 --warmup 2
```

### Save 결과

```bash
# JSON output (auto-detected from .json extension)
truthound benchmark run --suite ci -o results.json

# HTML report (auto-detected from .html extension)
truthound benchmark run --suite full -o report.html

# Explicit format (overrides auto-detection)
truthound benchmark run --suite ci -o results.dat --format json
```

!!! note "참고"
CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

    - CLI 명령 실행에서 JSON, `.json`, `benchmark compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - CLI 명령 실행에서 HTML, `.html`, `.htm`, `pip install truthound[reports]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
    - CLI 명령 실행에서 JSON, Other을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

!!! warning "참고"
CLI 명령 실행에서 HTML, Jinja2, Install을(를) 다루는 항목입니다:
    ```bash
    pip install truthound[reports]
    ```

Output 파일 (`results.json`):
```json
{
  "suite_name": "ci",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:15Z",
  "environment": {
    "python_version": "3.13.1",
    "polars_version": "1.x",
    "truthound_version": "1.x",
    "platform": "Darwin",
    "architecture": "arm64",
    "cpu_count": 10,
    "memory_total_gb": 36.0
  },
  "results": [
    {
      "benchmark_name": "profile",
      "benchmark_category": "profiling",
      "parameters": {
        "row_count": 10000,
        "iterations": 3
      },
      "metrics": {
        "timing": {
          "mean_seconds": 0.00072,
          "std_dev_seconds": 0.00011,
          "min_seconds": 0.00060,
          "max_seconds": 0.00090
        },
        "throughput": {
          "rows_per_second": 13840000
        }
      },
      "status": "passed"
    }
  ]
}
```

### Baseline Management

```bash
# Save results as baseline
truthound benchmark run --suite ci --save-baseline
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Baseline saved to: .benchmarks/baseline.json
```

```bash
# Compare against baseline
truthound benchmark run --suite ci --compare-baseline
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Benchmark Comparison
====================
Baseline: 2025-01-14 (ci suite)
Current:  2025-01-15 (ci suite)

Benchmark    Baseline    Current     Change      Status
───────────────────────────────────────────────────────────────────
profile      0.376s      0.382s      +1.6%       OK
check        0.524s      0.498s      -5.0%       IMPROVED
scan         0.245s      0.289s      +18.0%      REGRESSION
───────────────────────────────────────────────────────────────────

Status: REGRESSION DETECTED (1 benchmark)
Exit code: 1
```

### Verbose Output

```bash
truthound benchmark run profile --verbose
```

CLI 명령 실행에서 Shows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Use Cases

### 1. Development Testing

```bash
# Quick check during development
truthound benchmark run profile --size tiny --iterations 3
```

### 2. 성능 Optimization

```bash
# Detailed benchmark before optimization
truthound benchmark run --suite full -o before.json --format json

# ... make changes ...

# Compare after optimization
truthound benchmark compare before.json after.json
```

### 3. CI/CD 통합

```yaml
# GitHub Actions
- name: Performance Benchmark
  run: |
    truthound benchmark run --suite ci \
      --compare-baseline \
      --threshold 15.0
```

### 4. Size Scaling Analysis

```bash
# Test performance at different scales
for size in tiny small medium large; do
  truthound benchmark run profile \
    --size $size \
    -o "profile_${size}.json" \
    --format json
done
```

### 5. Stress Testing

```bash
# Large-scale stress test
truthound benchmark run --suite full \
  --size xlarge \
  --iterations 3 \
  -o stress_test.html \
  --format html
```

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Related Commands

- CLI 명령 실행에서 `benchmark list`, List을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [`benchmark compare`](compare.md) - Compare 벤치마크 결과

## 함께 보기

- [벤치마크 개요](index.md)
- [성능 Guide](../../guides/performance.md)

### 네이티브 실행과 저장 증거

`BenchmarkConfig`는 양수 측정 횟수, 음수가 아닌 준비 반복 횟수와 유한한
시간·메모리·cooldown 설정을 검증합니다. 시간·프로세스 RSS 검사는 setup과
반복 전후에 적용하며 실행 중인 단일 반복을 강제로 중단하지 않습니다. 실행
실패에도 teardown을 수행합니다. 실패하거나 빈 측정은 네이티브 성능 임계값과
비교를 통과할 수 없습니다. 상대 임계값은 일치하는 성공 기준값이 필요합니다.

저장된 `SuiteResult` 증거는 모든 원본 측정 시간, 사용자 지표, 환경 시각,
실행 시각, 매개변수와 태그를 포함합니다. 쓰기는 원자적 교체를 사용합니다.
읽기는 실제 결과를 복원하며 중복 JSON 키, 비유한 값, 반복 횟수 불일치와
64 MiB 초과 파일을 거부합니다. 원본 측정값이 없는 과거 요약 파일은 손실 없이
복원할 수 없어 비교 전에 다시 실행해야 하며, 비어 있는 성공 기준으로 바꾸지
않습니다. 기존 기준·이력 파일의 손상은 오류로 전파합니다. 이력 재로딩은 모든
파일을 검증한 후 메모리 목록을 교체합니다.

`RegressionDetector.save_baseline`은 현재 기준과 고유 시각의 이력 파일을
기록합니다. 각 교체는 원자적이지만 두 파일이 단일 트랜잭션은 아니므로 오류 시
한 파일만 기록될 수 있습니다. 저장된 기준은 증거이며 그 자체로 릴리스 수준
parity 결과가 되지 않습니다.

네이티브 데이터 생성기는 `GeneratorConfig`를 캡처·검증합니다. 0을 포함한
지정 시드는 기본 표 값·사용자 열·UUID를 제어합니다. 중복 주입 후에도
`row_count`는 출력 행 수를 유지하고 null 주입은 열 dtype을 보존합니다.
`ColumnSpec.null_ratio=None`은 전역 비율을 상속하며 명시적 0은 해당 열의
null을 끕니다. Sparse 패턴은 의도적으로 최소 80% null을 적용합니다.
Cardinality 제한은 null 주입 전에 적용합니다. 수치 패턴은 범위를 지키고
JSON·URL 열은 지정 형식의 값을 생성합니다. 지원하지 않는 타입·패턴·매개변수
조합은 관련 없는 문자열을 조용히 생성하지 않고 실패합니다.

`parallelize`는 사용자 표 열의 작업자(최대 4개)를 제어하며 같은 시드의
병렬·순차 출력은 일치합니다. `chunk_size`는 출력 청크 배치를 제어합니다.
고정 스키마 생성기는 네이티브 경로를 사용하고 사용자 열 명세는 표 생성기에
적용됩니다. 시간 간격은 시계열·금융 타임스탬프에 실제 적용되며 0을 포함한
요청 행 수를 유지합니다. 생성은 여전히 프레임을 구체화하며 청크 배치나
`generate_lazy()`가 제한된 메모리 스트리밍을 보장하지는 않습니다.

Suite 비교는 비어 있지 않은 고유 벤치마크 이름이 양쪽에서 일치해야 합니다. 누락·중복 측정은 부분 비교 성공으로 처리하지 않고 오류를 발생시킵니다. 비교 판정 임계값은 유한한 음수가 아닌 수여야 합니다.
