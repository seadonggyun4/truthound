# truthound 벤치마크 compare

CLI 명령 실행에서 Compare을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Synopsis

```bash
truthound benchmark compare <baseline> <current> [OPTIONS]
```

## Arguments

| CLI 명령 실행에서 Argument을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-------------|
| CLI 명령 실행에서 `baseline`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, Path을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `current`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 JSON, Path을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

!!! note "참고"
CLI 명령 실행에서 JSON, `benchmark run`, Both을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
CLI 명령 실행에서 `.json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```bash
    # Generate benchmark results (JSON format auto-detected from .json extension)
    truthound benchmark run --suite ci -o results.json
    ```

!!! warning "참고"
CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
CLI 명령 실행에서 CSV, Parquet을(를) 다루는 항목입니다:

    ```
    Warning: baseline file 'data.csv' does not have .json extension.
    This command compares benchmark result JSON files, not data files.

    Did you mean to run a benchmark first?
      truthound benchmark run --suite ci -o baseline.json
    ```

## Options

| CLI 명령 실행에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|---------|-------------|
| CLI 명령 실행에서 `--threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-t`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `10.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 성능 difference threshold (%) |
| CLI 명령 실행에서 `--format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `-f`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 `console`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Output을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Description

The `benchmark compare` command compares 벤치마크 결과:

1. CLI 명령 실행에서 Loads을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 Compares을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. CLI 명령 실행에서 Calculates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. **리포트** regressions and improvements
5. CLI 명령 실행에서 Returns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시

### Basic Comparison

```bash
truthound benchmark compare baseline.json current.json
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```
Benchmark Comparison
====================
Baseline: baseline.json (2025-01-14)
Current:  current.json (2025-01-15)
Threshold: 10.0%

Results
───────────────────────────────────────────────────────────────────
Benchmark       Baseline    Current     Change      Status
───────────────────────────────────────────────────────────────────
profile         0.376s      0.382s      +1.6%       OK
check           0.524s      0.498s      -5.0%       IMPROVED
scan            0.245s      0.289s      +18.0%      REGRESSION
compare         0.412s      0.425s      +3.2%       OK
───────────────────────────────────────────────────────────────────

Summary
───────────────────────────────────────────────────────────────────
Total Benchmarks: 4
Improved: 1 (check: -5.0%)
Regressed: 1 (scan: +18.0%)
Unchanged: 2

Status: REGRESSION DETECTED
The following benchmarks exceeded the 10.0% threshold:
  - scan: +18.0% (0.245s → 0.289s)
───────────────────────────────────────────────────────────────────
```

### Custom Threshold

```bash
# Stricter threshold (5%)
truthound benchmark compare baseline.json current.json --threshold 5.0

# More lenient threshold (20%)
truthound benchmark compare baseline.json current.json --threshold 20.0
```

### JSON Output

```bash
truthound benchmark compare baseline.json current.json --format json
```

CLI 명령 실행에서 Output을(를) 다루는 항목입니다:
```json
{
  "baseline_file": "baseline.json",
  "current_file": "current.json",
  "threshold_percent": 10.0,
  "regressions": [
    {
      "name": "scan",
      "baseline_seconds": 0.245,
      "current_seconds": 0.289,
      "change_percent": 18.0
    }
  ],
  "improvements": [
    {
      "name": "check",
      "baseline_seconds": 0.524,
      "current_seconds": 0.498,
      "change_percent": -5.0
    }
  ],
  "has_regressions": true
}
```

## Comparison Status

| CLI 명령 실행에서 Status을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-----------|-------------|
| CLI 명령 실행에서 `OK`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Change을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Within을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 `IMPROVED`, IMPROVED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Change을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 성능 improved |
| CLI 명령 실행에서 `REGRESSION`, REGRESSION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Change을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 성능 degraded |

## Use Cases

### 1. CI/CD Regression Detection

```yaml
# GitHub Actions
- name: Run Benchmarks
  run: truthound benchmark run --suite ci -o current.json --format json

- name: Check for Regression
  run: |
    truthound benchmark compare \
      benchmarks/baseline.json \
      current.json \
      --threshold 10.0
```

### 2. Before/After Optimization

```bash
# Before optimization
truthound benchmark run --suite full -o before.json --format json

# ... make code changes ...

# After optimization
truthound benchmark run --suite full -o after.json --format json

# Compare
truthound benchmark compare before.json after.json
```

### 3. 릴리스 검증

```bash
# Compare against last release
truthound benchmark compare \
  releases/v1.0.0_benchmark.json \
  releases/v1.1.0_benchmark.json \
  --threshold 5.0
```

### 4. Daily 성능 Tracking

```bash
#!/bin/bash
# daily_benchmark.sh
TODAY=$(date +%Y%m%d)
YESTERDAY=$(date -d "yesterday" +%Y%m%d)

# Run today's benchmark
truthound benchmark run --suite ci -o "benchmarks/${TODAY}.json" --format json

# Compare with yesterday
if [ -f "benchmarks/${YESTERDAY}.json" ]; then
  truthound benchmark compare \
    "benchmarks/${YESTERDAY}.json" \
    "benchmarks/${TODAY}.json" \
    --threshold 15.0
fi
```

## Comparison Algorithm

1. CLI 명령 실행에서 Match을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. CLI 명령 실행에서 `((current - baseline) / baseline) * 100`, Calculate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. CLI 명령 실행에서 Apply, Mark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. **리포트 status**: OK, IMPROVED, or REGRESSION

## Threshold Guidelines

| CLI 명령 실행에서 Threshold을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Strict을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Standard, CI/CD 파이프라인 |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Development을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Lenient을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Exit Codes

| CLI 명령 실행에서 Code을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Condition을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Success을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | CLI 명령 실행에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

!!! note "참고"
CLI 명령 실행에서 Exit, CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
CLI 명령 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Commands

- CLI 명령 실행에서 `benchmark run`, Run을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- CLI 명령 실행에서 `benchmark list`, List을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- [벤치마크 개요](index.md)
- [CI/CD 통합](../../guides/ci-cd.md)
